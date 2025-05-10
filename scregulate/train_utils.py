import numpy as np
import torch

def schedule_parameter(epoch, start_value, max_value, total_epochs, scale=0.6):
    """
    Schedules a parameter (e.g., alpha, beta, gamma) to grow from `start_value` to `max_value`
    over a given fraction of the training epochs.

    Args:
        epoch: Current epoch.
        start_value: Starting value of the parameter.
        max_value: Maximum value of the parameter.
        total_epochs: Total number of training epochs.
        scale: Fraction of epochs over which the parameter reaches `max_value`.

    Returns:
        Scheduled parameter value for the current epoch.
    """
    scaled_epochs = total_epochs * scale
    if epoch >= scaled_epochs:
        return max_value
    return start_value + (max_value - start_value) * (epoch / scaled_epochs)


def schedule_mask_factor(epoch, freeze_epochs):
    """
    Computes the mask factor using a sigmoidal schedule.

    Args:
        epoch: Current epoch.
        freeze_epochs: Epoch at which the mask fully applies.

    Returns:
        Mask factor value for the current epoch.
    """
    return 1.0 / (1.0 + np.exp((epoch - freeze_epochs // 2) / (freeze_epochs // 20)))


def apply_gradient_mask(linear_layer, mask_float, mask_factor):
    """
    Applies a gradient mask to a layer's weights.

    Args:
        linear_layer: Linear layer whose gradients are masked.
        mask_float: Precomputed mask (e.g., `W_prior == 0`).
        mask_factor: Factor controlling the strength of the mask.
    """
    with torch.no_grad():
        linear_layer.weight.grad.mul_(1 - mask_factor * mask_float)


def compute_average_loss(total_loss, total_samples):
    """
    Computes the average loss per sample.

    Args:
        total_loss: Total accumulated loss over an epoch.
        total_samples: Total number of samples in the epoch.

    Returns:
        Average loss per sample.
    """
    return total_loss.item() / total_samples


def clip_gradients(model, max_norm=0.5):
    """
    Clips gradients of the model to prevent exploding gradients.

    Args:
        model: PyTorch model.
        max_norm: Maximum allowed norm for gradients.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
