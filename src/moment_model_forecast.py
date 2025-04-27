import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from tqdm import tqdm
from momentfm import MOMENTPipeline
from momentfm.utils.forecasting_metrics import get_forecasting_metrics
from momentfm import MOMENTPipeline
import torch.nn as nn

device = 'cpu'

# Custom Dataset Class for Multivariate Time Series
class TimeSeriesDataset(Dataset):
    def __init__(self, data, forecast_horizon=192, context_length=512, target_column=None):
        self.data = data
        self.forecast_horizon = forecast_horizon
        self.context_length = context_length
        self.target_column = target_column
        
        # Normalize the data (scale features to the range [0, 1])
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_scaled = self.scaler.fit_transform(data)
    
    def __len__(self):
        return len(self.data) - self.context_length - self.forecast_horizon
    
    def __getitem__(self, idx):
        # Input sequence (context length)
        x = self.data_scaled[idx:idx+self.context_length]
        
        # Output sequence (forecast horizon)
        y = self.data_scaled[idx+self.context_length:idx+self.context_length+self.forecast_horizon]
        
        # Return both sequences
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Load your multivariate time series dataset (6 columns) from CSV
df = pd.read_csv('data/processed/cleaned_train_set.csv', parse_dates=True, index_col=0)  # Assume first column is time index

# Inspect the first few rows of the dataset
print(df.head())

# Ensure the data has 6 columns (multivariate)
assert df.shape[1] == 6, "Dataset must have 6 columns for multivariate forecasting."

# Convert the data to numpy array
data = df.values

# Forecasting parameters
forecast_horizon = 192  # Number of time steps to forecast
context_length = 512    # Length of the historical context

# Create the Dataset and DataLoader
train_dataset = TimeSeriesDataset(data, forecast_horizon=forecast_horizon, context_length=context_length)
test_size = int(0.2 * len(train_dataset))  # 20% for testing
train_size = len(train_dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': forecast_horizon,
        'patch_len': 8,  # Ensure this matches your input size (6 features)
        'head_dropout': 0.1,
        'weight_decay': 0,
        'freeze_encoder': True,
        'freeze_embedder': True,
        'freeze_head': False,
    },
    strict=False,
    ignore_mismatched_sizes=True
)

# Adjust the model layers to match 6 features (modify the number of features if needed)
model.patch_embedding.value_embedding = nn.Linear(6, 1024)  # Update for 6 features (input size)
model.head.linear = nn.Linear(1024, 6)  # Ensure output layer matches your forecast horizon

model.init()
model = model.to('cpu')  # Use the correct device (CPU or GPU)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Create learning rate scheduler
max_lr = 1e-4
total_steps = len(train_loader) * 1  # Training for 1 epoch
scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3)

# Training loop
cur_epoch = 0
max_epoch = 1  # Set to the number of epochs needed for fine-tuning

while cur_epoch < max_epoch:
    losses = []
    model.train()  # Set the model to training mode
    for timeseries, forecast in tqdm(train_loader, total=len(train_loader)):
        timeseries = timeseries.float().to(device)  # Use the correct device
        forecast = forecast.float().to(device)  # Use the correct device

        # Forward pass
        output = model(x_enc=timeseries)  # Assuming output contains a 'forecast' attribute
        
        # Ensure the forecast is extracted from the model's output (depending on your model's architecture)
        loss = criterion(output['forecast'], forecast)

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())

    avg_loss = np.mean(losses)
    print(f"Epoch {cur_epoch}: Train loss: {avg_loss:.3f}")
    scheduler.step()  # Update learning rate
    cur_epoch += 1

    # Evaluate the model on the test split
    model.eval()  # Set the model to evaluation mode
    trues, preds, test_losses = [], [], []
    with torch.no_grad():
        for timeseries, forecast in tqdm(test_loader, total=len(test_loader)):
            timeseries = timeseries.float().to(device)  # Use the correct device
            forecast = forecast.float().to(device)  # Use the correct device

            output = model(x_enc=timeseries)  # Again, extract forecast correctly
            
            loss = criterion(output['forecast'], forecast)
            test_losses.append(loss.item())

            trues.append(forecast.detach().cpu().numpy())
            preds.append(output['forecast'].detach().cpu().numpy())

    avg_test_loss = np.mean(test_losses)
    print(f"Epoch {cur_epoch}: Test loss: {avg_test_loss:.3f}")

    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)

    # Metrics (optional, add forecasting metrics calculation)
    metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')
    print(f"Epoch {cur_epoch}: Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f}")

# Visualization: Visualize forecast vs. ground truth for a sample
channel_idx = np.random.randint(0, 6)  # 6 channels in this dataset
time_index = np.random.randint(0, trues.shape[0])

history = trues[time_index, channel_idx, :]
pred = preds[time_index, channel_idx, :]

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(range(len(history)), history, label='History', c='darkblue')
plt.plot(range(len(history), len(history) + len(pred)), pred, label='Forecast', color='red', linestyle='--')

plt.title(f"Forecasting Results -- (idx={time_index}, channel={channel_idx})", fontsize=18)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.legend(fontsize=14)
plt.show()