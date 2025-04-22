import torch
import numpy as np
import pandas as pd
import scanpy as sc
from torch.utils.data import DataLoader
import gc
import random

# ---------- Random Seed Utility ----------
def set_random_seed(seed=42):
    """
    Sets the random seed for reproducibility across multiple libraries.

    Args:
        seed (int): Random seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disables benchmarking to avoid randomness
set_random_seed()

# ---------- Data Loading/Handling ----------
def as_GTRD(net):
    W_prior_df = net.copy()
    collectri_format = W_prior_df.stack().reset_index()
    collectri_format.columns = ["source", "target", "weight"].copy()
    collectri_format = collectri_format[collectri_format["weight"] == 1]
    return collectri_format.reset_index(drop=True)
    
def as_TF_Link(net):
    W_prior_df = net.copy()
    collectri_format = W_prior_df[['Name.TF', 'Name.Target']]
    collectri_format.columns = ["source", "target"]
    collectri_format = collectri_format.assign(weight=1)
    collectri_format = collectri_format.drop_duplicates()
    return collectri_format.reset_index(drop=True)
    
def create_dataloader(scRNA_batch, batch_size, shuffle=False, num_workers=0, collate_fn=None):
    """
    Creates a PyTorch DataLoader for the scRNA_batch data.

    Args:
        scRNA_batch (torch.Tensor): Input data for training.
        batch_size (int): Batch size for training.
        shuffle (bool): Whether to shuffle the data during loading.
        num_workers (int): Number of workers for data loading.
        collate_fn (callable, optional): Custom collate function for DataLoader.

    Returns:
        DataLoader: PyTorch DataLoader for the scRNA_batch data.
    """
    return DataLoader(scRNA_batch, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                      collate_fn=collate_fn)  # Pass custom collate_fn if provided)

def clear_memory(device):
    """
    Clears memory for the specified device.

    Args:
        device (str): The device to clear memory for ('cuda' or 'cpu').
    """
    if device == "cuda":
        torch.cuda.empty_cache()
    else:
        gc.collect()

def extract_GRN(adata, matrix_type="posterior"):
    """
    Extract the GRN matrix (either prior or posterior) from an AnnData object as a pandas DataFrame.

    Args:
        adata (AnnData): The AnnData object containing GRN information in `.uns`.
        matrix_type (str): The type of GRN matrix to extract, either 'prior' or 'posterior'.
                           Defaults to 'posterior'.

    Returns:
        pd.DataFrame: A DataFrame where rows are genes and columns are transcription factors (TFs).
    """
    # Map short names to full keys
    matrix_key_map = {
        "posterior": "GRN_posterior",
        "prior": "GRN_prior"
    }

    if matrix_type not in matrix_key_map:
        raise ValueError(f"Invalid matrix_type '{matrix_type}'. Choose from 'prior' or 'posterior'.")

    full_key = matrix_key_map[matrix_type]

    if full_key not in adata.uns:
        raise ValueError(f"{full_key} is not stored in the provided AnnData object.")

    grn_data = adata.uns[full_key]
    matrix = grn_data["matrix"]
    TF_names = grn_data["TF_names"]
    gene_names = grn_data["gene_names"]

    # Determine the orientation of the matrix
    if matrix.shape[0] == len(TF_names) and matrix.shape[1] == len(gene_names):
        # TFs as rows, genes as columns (needs transposition)
        GRN_df = pd.DataFrame(matrix.T, index=gene_names, columns=TF_names)
    elif matrix.shape[0] == len(gene_names) and matrix.shape[1] == len(TF_names):
        # Genes as rows, TFs as columns (already correct orientation)
        GRN_df = pd.DataFrame(matrix, index=gene_names, columns=TF_names)
    else:
        raise ValueError("Matrix dimensions do not match the lengths of `gene_names` and `TF_names`.")

    return GRN_df


def to_torch_tensor(matrix, device="cpu"):
    """
    Converts a dense or sparse matrix to a PyTorch tensor.

    Args:
        matrix: Numpy array or sparse matrix.
        device (str): Device to load the tensor onto.

    Returns:
        torch.Tensor: Dense or sparse tensor.
    """
    if isinstance(matrix, np.ndarray):
        return torch.tensor(matrix, dtype=torch.float32).to(device)
    elif hasattr(matrix, "tocoo"):  # Sparse matrix support
        coo = matrix.tocoo()
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape).to(device)
    else:
        raise TypeError("Unsupported matrix type for conversion to PyTorch tensor.")


# ---------- Parameter Scheduling ----------
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
    Computes the mask factor using a sigmoid function.

    Args:
        epoch (int): Current epoch.
        freeze_epochs (int): Epoch at which the mask factor reaches its midpoint.

    Returns:
        float: Mask factor for the current epoch.
    """
    return 1.0 / (1.0 + np.exp((epoch - freeze_epochs // 2) / (freeze_epochs // 20)))


# ---------- Gradient Operations ----------
def apply_gradient_mask(layer, mask, mask_factor):
    """
    Applies a gradient mask to a layer's weights.

    Args:
        layer (torch.nn.Linear): Linear layer whose gradients need masking.
        mask (torch.Tensor): Binary mask indicating where gradients should be scaled.
        mask_factor (float): Scaling factor for the mask.

    Returns:
        None
    """
    with torch.no_grad():
        layer.weight.grad.mul_(1 - mask_factor * mask)


def clip_gradients(model, max_norm=0.5):
    """
    Clips the gradients of the model parameters.

    Args:
        model (torch.nn.Module): PyTorch model.
        max_norm (float): Maximum norm for gradient clipping.

    Returns:
        None
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


# ---------- Loss Utility ----------
def compute_average_loss(total_loss, total_samples):
    """
    Computes the average loss per sample.

    Args:
        total_loss (torch.Tensor): Total accumulated loss.
        total_samples (int): Total number of samples processed.

    Returns:
        float: Average loss per sample.
    """
    return total_loss.item() / total_samples


# ---------- Modality Enhancement ----------
def set_active_modality(adata, modality_name):
    """
    Sets the active modality in the AnnData object.

    Args:
        adata (AnnData): Parent AnnData object containing multiple modalities.
        modality_name (str): The name of the modality to activate.

    Returns:
        AnnData: A new AnnData object with the specified modality set as the active one.
    """
    # Validate input modality
    if modality_name not in adata.modality:
        raise ValueError(f"Modality '{modality_name}' not found in AnnData.modality.")

    # Retrieve selected modality
    selected_modality = adata.modality[modality_name]

    # Create a new AnnData object with the selected modality
    temp_adata = sc.AnnData(
        X=selected_modality.X,
        obs=selected_modality.obs.copy(),
        var=selected_modality.var.copy(),
        obsm=selected_modality.obsm.copy(),
        uns=adata.uns.copy(),  # Retain shared metadata in `uns`
    )

    # Retain modality-specific `.obsm` entries
    temp_adata.obsm.update(adata.obsm)

    # Attach all modalities back to the new AnnData object
    temp_adata.modality = adata.modality

    return temp_adata

def extract_modality(adata, modality_name):
    """
    Extract a specific modality as an independent AnnData object.

    Args:
        adata (AnnData): The AnnData object containing multiple modalities.
        modality_name (str): The name of the modality to extract.

    Returns:
        AnnData: The requested modality as a standalone AnnData object.
    """
    # Ensure the modality exists
    if not hasattr(adata, "modality") or modality_name not in adata.modality:
        raise ValueError(f"Modality '{modality_name}' not found in the AnnData object.")

    # Extract the requested modality
    selected_modality = adata.modality[modality_name]
    return sc.AnnData(
        X=selected_modality.X.copy(),
        obs=selected_modality.obs.copy(),
        var=selected_modality.var.copy(),
        obsm=selected_modality.obsm.copy(),
        uns=selected_modality.uns.copy(),
        layers=selected_modality.layers.copy() if hasattr(selected_modality, "layers") else None
    )
