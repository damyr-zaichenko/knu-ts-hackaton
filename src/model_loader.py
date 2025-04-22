import os
import os.path as p

def load_timesfm_model():
    """Loads and initializes the TimesFM model with specified hyperparameters and checkpoint."""
    
    # Set directory to store model
    base_dir = p.abspath(p.dirname(__file__))
    model_cache_dir = p.join(base_dir, "..", "models", "timesfm")
    os.environ["HF_HOME"] = model_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = model_cache_dir
    os.environ["TORCH_HOME"] = model_cache_dir

    import timesfm

    # Init and return model
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=128,
            num_layers=50,
            use_positional_embedding=False,
            context_len=2048,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )
    return model