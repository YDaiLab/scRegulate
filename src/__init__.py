# src/__init__.py

# Core training and auto-tuning
from .train import train_model, adapt_prior_and_data
from .auto_tuning import auto_tune

# Model architecture
from .vae_model import scRNA_VAE

# Loss functions and gradient norm
from .loss_functions import loss_function, get_gradient_norm

# Training utilities
from .train_utils import (
    schedule_parameter,
    schedule_mask_factor,
    apply_gradient_mask,
    compute_average_loss,
    clip_gradients,
)

# General utilities
from .utils import (
    set_random_seed,
    create_dataloader,
    clear_memory,
    extract_GRN,
    to_torch_tensor,
    set_active_modality,
    extract_modality,
)

