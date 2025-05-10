import torch
import torch.nn.functional as F

def loss_function(recon_x, x, mu, logvar):
    """
    Computes the loss function for the Variational Autoencoder (VAE).
    The loss consists of:
    - Mean Squared Error (MSE) between the input and reconstructed data.
    - Kullback-Leibler Divergence (KLD) to regularize the latent space.

    Args:
        recon_x: Reconstructed input.
        x: Original input.
        mu: Mean vector from the encoder.
        logvar: Log variance vector from the encoder.

    Returns:
        MSE: Reconstruction loss.
        KLD: KL divergence loss.
    """
    MSE = F.mse_loss(recon_x, x, reduction='sum')  # Reconstruction loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence
    return MSE, KLD


def get_gradient_norm(model, norm_type=2):
    """
    Computes the total gradient norm for a model.

    Args:
        model: The PyTorch model.
        norm_type: The type of norm to compute (default is 2 for Euclidean norm).

    Returns:
        total_norm: The total gradient norm.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm
