import copy
import torch
import numpy as np
import scanpy as sc
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import logging
import torch.nn.functional as F
from .loss_functions import loss_function
from .utils import (
    set_random_seed,  # To ensure reproducibility
    create_dataloader,  # For DataLoader creation
    clear_memory,       # To handle memory cleanup
    to_torch_tensor,    # Converts dense or sparse data to PyTorch tensors
    schedule_parameter, # For parameter scheduling during training
)
set_random_seed()

# ---------- Logging Configuration ----------
finetune_logger = logging.getLogger("finetune")
finetune_logger.setLevel(logging.INFO)

# Reset logger: Remove any existing handlers
while finetune_logger.handlers:
    finetune_logger.removeHandler(finetune_logger.handlers[0])

# Define a new handler
handler = logging.StreamHandler()  # Logs to console; use FileHandler for a file
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
finetune_logger.addHandler(handler)

finetune_logger.propagate = False  # Prevent duplicate logs from root logger

# ---------- Sparsity Handling ----------
def sparse_collate_fn(batch):
    """
    Custom collate function to handle sparse tensors.

    Args:
        batch: A batch of data (potentially sparse).

    Returns:
        torch.Tensor: Dense tensor batch.
    """
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch)
    elif isinstance(batch[0], torch.sparse_coo_tensor):
        # Convert sparse tensors to dense format before stacking
        return torch.stack([b.to_dense() for b in batch])
    else:
        raise TypeError(f"Unsupported batch type: {type(batch[0])}")

# ---------- Extract TF Activities ----------
def extract_tf_activities_deterministic(model, data_loader, device="cuda"):
    """
    Extract transcription factor (TF) activities deterministically using the trained model.

    Args:
        model (torch.nn.Module): The trained model to use for inference.
        data_loader (DataLoader): DataLoader containing the cluster-specific data.
        device (str): Device for computation ('cuda' or 'cpu'). Default is 'cuda'.

    Returns:
        np.ndarray: Matrix of inferred TF activities.
    """
    model.eval()
    tf_activities = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)  # Use mean (mu) for deterministic encoding
            tf_activity = model.decode(mu).clamp(min=0)
            #tf_activity = model.decode(mu).clamp(min=0, max=5.0)
            tf_activities.append(tf_activity.cpu().numpy())
    return np.concatenate(tf_activities, axis=0)

# ---------- Cluster-sepcific handling ----------
def scale_and_format_grns(W_posteriors, gene_names, tf_names):
    """
    Apply MinMax scaling to each W matrix (per cluster) and return a dict of DataFrames.
    """
    scaled_grns = {}
    for cluster, W in W_posteriors.items():
        scaled = MinMaxScaler().fit_transform(W)
        scaled_df = pd.DataFrame(scaled, index=gene_names, columns=tf_names)
        scaled_grns[cluster] = scaled_df

    return scaled_grns

def finalize_fine_tuning_outputs(processed_adata, all_tf_activities_matrix, tf_names, W_posteriors, gene_names):
    """
    Prepares and returns the processed AnnData, TF activity AnnData, and GRNs.
    """
    # Attach TF activities to AnnData
    processed_adata.obsm["TF_finetuned"] = all_tf_activities_matrix
    processed_adata.uns["W_posteriors_per_cluster"] = W_posteriors

    # Create new AnnData object for TF activities
    tf_activities_adata = sc.AnnData(
        X=all_tf_activities_matrix,
        obs=processed_adata.obs.copy(),
        var=pd.DataFrame(index=tf_names),
    )

    # Format GRNs
    scaled_grns = scale_and_format_grns(
        W_posteriors, gene_names=gene_names, tf_names=tf_names
    )

    return processed_adata, tf_activities_adata, scaled_grns



# ---------- Fine-Tuning for All Clusters ----------
def fine_tune_clusters(
    processed_adata,
    model,
    cluster_key=None,
    epochs=500,
    batch_size=3500,
    device=None,
    max_weight_norm=4,
    log_interval=100,
    early_stopping_patience=250,
    min_epochs=0,
    beta_start=0,
    beta_max=0.5,
    tf_mapping_lr=4e-04,  # Learning rate for tf_mapping layer
    fc_output_lr=2e-05,   # Learning rate for fc_output layer
    default_lr=3.5e-05,     # Default learning rate for other layers
    verbose=True,
):
    """
    Fine-tune the model for each cluster in the AnnData object with logging intervals and early stopping.

    Args:
        processed_adata (AnnData): AnnData object containing RNA data and cluster annotations.
        model (torch.nn.Module): Pre-trained model to fine-tune.
        cluster_key (str): Key in `.obs` containing cluster annotations. Default is None.
        epochs (int): Number of epochs for fine-tuning each cluster. Default is 5000.
        batch_size (int): Batch size for fine-tuning. Default is 2000.
        device (str): Device to perform computation ('cuda' or 'cpu'). Default is 'cuda'.
        beta_start (float): Initial value for beta warm-up. Default is 0.01.
        beta_max (float): Maximum value for beta warm-up. Default is 0.1.
        max_weight_norm (float): Maximum weight norm for clipping. Default is 0.1.
        log_interval (int): Number of epochs between logging updates. Default is 100.
        early_stopping_patience (int): Number of epochs to wait without improvement. Default is 1000.
        min_epochs (int): Minimum number of epochs before checking for early stopping. Default is 1000.
        tf_mapping_lr (float): Learning rate for the tf_mapping layer. Default is 1e-2.
        fc_output_lr (float): Learning rate for the fc_output layer. Default is 1e-3.
        default_lr (float): Learning rate for other layers. Default is 1e-4.
        verbose (bool): If True, enables detailed logging.

    Returns:
        AnnData: Updated AnnData object with fine-tuned TF activities (`obsm`) and cluster-specific W_posteriors (`uns`).

    Notes:
        - Do not change the learning rates unless you are an expert.
        - tf_mapping_lr (1e-2): Faster learning rate for tf_mapping layer.
        - fc_output_lr (1e-3): Slower learning rate for fc_output layer.
        - default_lr (1e-4): Learning rate for any other trainable layers.
    """
    set_random_seed()
    batch_size = int(batch_size)
    
    # Set logging verbosity, control verbosity of train_model logger only.
    if verbose:
        finetune_logger.setLevel(logging.INFO)
    else:
        finetune_logger.setLevel(logging.WARNING)

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    finetune_logger.info("Starting fine-tuning for cluster(s)...")

    # Align gene indices to match GRN dimensions
    grn_gene_names = processed_adata.uns["GRN_posterior"]["gene_names"]
    adata_gene_names = processed_adata.var.index
    gene_indices = [adata_gene_names.get_loc(gene) for gene in grn_gene_names if gene in adata_gene_names]

    
    finetune_logger.info(f"Aligning data to {len(gene_indices)} genes matching the GRN.")

    if not cluster_key:
        finetune_logger.info("Cluster key not provided, fine-tuning on all cells together...")
        cluster_key = 'all'
        processed_adata.obs[cluster_key] = 1
        
    unique_clusters = processed_adata.obs[cluster_key].unique()
    original_model = copy.deepcopy(model)
    W_posteriors_per_cluster = {}
    all_tf_activities = []
    cluster_indices_list = []
    total_cells = len(processed_adata)
    for cluster in unique_clusters:
        if len(unique_clusters)>1:
            finetune_logger.info(f"Fine-tuning {cluster} for {epochs} epochs...")
        else:
            finetune_logger.info(f"Fine-tuning on all cells for {epochs} epochs...")
            
        cluster_size = len(processed_adata.obs[processed_adata.obs[cluster_key] == cluster])
        proportional_epochs = max(min_epochs, int(epochs * (cluster_size / total_cells)))

        # Extract cluster data and ensure dense format
        cluster_indices = processed_adata.obs[cluster_key] == cluster
        cluster_data = processed_adata[cluster_indices].X[:, gene_indices]

        # Convert sparse to dense if needed
        if not isinstance(cluster_data, np.ndarray):
            cluster_data = cluster_data.todense()
        cluster_tensor = to_torch_tensor(cluster_data, device=device)

        # Create DataLoader
        cluster_loader = create_dataloader(cluster_tensor, batch_size=batch_size)

        # Fine-tune model for the cluster with early stopping
        model_copy = copy.deepcopy(original_model)
        best_loss, epochs_no_improve = float("inf"), 0

        # Optimizer with configurable learning rates
        optimizer = torch.optim.AdamW([
            {"params": model_copy.tf_mapping.parameters(), "lr": tf_mapping_lr},  # Learning rate for tf_mapping
            {"params": model_copy.fc_output.parameters(), "lr": fc_output_lr},   # Learning rate for fc_output
            {"params": [p for n, p in model_copy.named_parameters() 
                        if "tf_mapping" not in n and "fc_output" not in n], "lr": default_lr}  # Default LR
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", 
                                                               patience=early_stopping_patience//2, 
                                                               factor=0.5, min_lr=default_lr)
        for epoch in range(proportional_epochs):
            model_copy.train()
            total_loss, total_samples = 0, 0
            # Warm-up beta
            beta = schedule_parameter(epoch, beta_start, beta_max, epochs)

            for batch in cluster_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                # Forward pass
                recon_batch, mu, logvar = model_copy(batch)
                MSE, _ = loss_function(recon_batch, batch, mu, logvar)
                l1_reg = torch.sum(torch.abs(model_copy.tf_mapping.weight))

                cluster_weight = len(cluster_indices) / len(processed_adata)
                loss = (MSE + beta * l1_reg)# * cluster_weight

                # Backward pass
                loss.backward()
                optimizer.step()

                # Weight clipping
                with torch.no_grad():
                    model_copy.tf_mapping.weight.clamp_(-max_weight_norm, max_weight_norm)

                total_loss += loss.item() * len(batch)
                total_samples += len(batch)

            # Average loss for the epoch
            avg_loss = total_loss / total_samples
            scheduler.step(avg_loss)

            # Logging
            if epoch % log_interval == 0 or epoch == epochs - 1:
                finetune_logger.info(f"Epoch {epoch+1}, Avg Loss: {avg_loss/processed_adata.shape[1]:.4f}")

            # Early stopping
            if epoch > min_epochs:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    finetune_logger.info(f"Early stopping for cluster {cluster} at epoch {epoch+1}")
                    break

        # Extract W_posterior and TF activities
        W_posterior_cluster = model_copy.tf_mapping.weight.detach().cpu().numpy()
        W_posteriors_per_cluster[cluster] = W_posterior_cluster

        tf_activities = extract_tf_activities_deterministic(model_copy, cluster_loader, device=device)

        cluster_indices_list.append(np.where(cluster_indices)[0])
        all_tf_activities.append(tf_activities)

        # Clear memory
        clear_memory(device)

    # [Modification 3] - We now build the final matrix in the ORIGINAL cell order
    n_cells = processed_adata.n_obs

    # If there's at least one cluster, get the size of the TF dimension from the first cluster's result
    n_tfs = all_tf_activities[0].shape[1] if len(all_tf_activities) > 0 else 0
    
    # [Modification 4] - Create a zeroed matrix to hold TF activities for ALL cells
    all_tf_activities_matrix = np.zeros((n_cells, n_tfs), dtype=np.float32)

    # [Modification 5] - Insert each cluster's activities into the correct rows
    for row_indices, cluster_acts in zip(cluster_indices_list, all_tf_activities):
        all_tf_activities_matrix[row_indices, :] = cluster_acts

    # Attach to adata
    processed_adata.obsm["TF_finetuned"] = all_tf_activities_matrix
    processed_adata.uns["W_posteriors_per_cluster"] = W_posteriors_per_cluster

    tf_names = processed_adata.modality['TF'].var.index
    
    tf_activities_adata = sc.AnnData(
        X=all_tf_activities_matrix,
        obs=processed_adata.obs.copy(),
        var=pd.DataFrame(index=tf_names),
    )

    gene_names = processed_adata.uns['GRN_posterior']['gene_names']
    tf_names = processed_adata.uns['GRN_posterior']['TF_names']
    
    processed_adata, tf_activities_adata, scaled_grns = finalize_fine_tuning_outputs(
        processed_adata,
        all_tf_activities_matrix,
        tf_names=tf_names,
        W_posteriors=W_posteriors_per_cluster,
        gene_names=gene_names,
    )

    # If only one GRN (no cluster_key given), return a single DataFrame
    if len(scaled_grns) == 1:
        scaled_grns = list(scaled_grns.values())[0]
    
    finetune_logger.info("Fine-tuning completed for all clusters.")
    return processed_adata, tf_activities_adata, model_copy, scaled_grns


