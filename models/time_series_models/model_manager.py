class TimeSeriesModelManager:
    def __init__(self, models: dict):
        self.models = models

    def fit_all(self, series):
        for model in self.models.values():
            model.fit(series)

    def forecast_all(self, steps):
        return {name: model.forecast(steps) for name, model in self.models.items()}