import time
import torch
import pandas as pd
import numpy as np
import scanpy as sc
import gc
import logging
from . import ulm_standalone as ulm
from .vae_model import scRNA_VAE
from .loss_functions import loss_function
from sklearn.preprocessing import MinMaxScaler
from .utils import (
    create_dataloader,
    schedule_parameter,
    schedule_mask_factor,
    apply_gradient_mask,
    clip_gradients,
    compute_average_loss,
    clear_memory,
    set_random_seed,
)
set_random_seed()

# Initialize logging
# Dedicated logger for train_model
# Reset logging to avoid duplicate handlers
train_logger = logging.getLogger("train_model")
train_logger.setLevel(logging.INFO)  # Default level

# Remove any existing handlers
for handler in train_logger.handlers[:]:
    train_logger.removeHandler(handler)

# Create new handler
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s - %(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
train_logger.addHandler(handler)

train_logger.propagate = False  # Prevent duplicate logs from parent loggers

def adapt_prior_and_data(rna_data, net, min_targets=10, min_TFs=1):
    """
    Adapt the RNA data and network to create W_prior.

    Args:
        rna_data (AnnData): AnnData object containing RNA data and metadata.
        net (pd.DataFrame): DataFrame containing 'source', 'target', and 'weight' columns.
        min_targets (int): Minimum number of target genes per TF to retain (default: 10).
        min_TFs (int): Minimum number of TFs per target gene to retain (default: 1).

    Returns:
        tuple: (W_prior (torch.Tensor), gene_names (list), TF_names (list))
    """
    train_logger.info("Adapting prior and data...")
    train_logger.info(f"Initial genes in RNA data: {rna_data.var.index.size}")

    # Validate network structure
    required_columns = {'source', 'target', 'weight'}
    if not required_columns.issubset(net.columns):
        train_logger.error(f"Network DataFrame is missing columns: {required_columns - set(net.columns)}")
        raise ValueError(f"The 'net' DataFrame must contain: {required_columns}")

    # Create binary matrix
    binary_matrix = pd.crosstab(net['source'], net['target'], values=net['weight'], aggfunc='first').fillna(0)

    # Filter genes in RNA data
    keep_genes = rna_data.var.index.intersection(net['target'])
    if keep_genes.empty:
        train_logger.error("No overlapping genes between RNA data and network.")
        raise ValueError("No overlapping genes between RNA data and network.")
    rna_data._inplace_subset_var(keep_genes)
    train_logger.info(f"Genes retained after intersection with network: {len(keep_genes)}")
    train_logger.info(f"Initial TFs in GRN matrix: {binary_matrix.index.size}")

    # Align binary matrix with RNA data
    binary_matrix = binary_matrix.loc[:, binary_matrix.columns.intersection(rna_data.var.index)]
    if binary_matrix.empty:
        train_logger.error("Binary matrix is empty after alignment.")
        raise ValueError("Binary matrix empty after alignment with RNA data.")

    # Remove TFs with no targets and genes with no TFs
    binary_matrix = binary_matrix.loc[binary_matrix.sum(axis=1) >= min_targets, :]
    binary_matrix = binary_matrix.loc[:, binary_matrix.sum(axis=0) >= min_TFs]
    if binary_matrix.empty:
        train_logger.error(f"Binary matrix empty after filtering (min_targets={min_targets}, min_TFs={min_TFs}).")
        raise ValueError("Binary matrix empty after filtering for min_targets and min_TFs.")

    # Convert to W_prior tensor
    W_prior = torch.tensor(binary_matrix.values, dtype=torch.float32).T

    # Metadata
    gene_names = binary_matrix.columns.tolist()
    TF_names = binary_matrix.index.tolist()

    train_logger.info(f"Retained {len(gene_names)} genes and {len(TF_names)} transcription factors.")

    return W_prior, gene_names, TF_names

def train_model(
    rna_data, net, 
    encode_dims=None, decode_dims=None, z_dim=40, 
    train_val_split_ratio=0.8, random_state=42,
    batch_size=3500, epochs=2000, freeze_epochs=1500, learning_rate=2.5e-4,
    alpha_start=0, alpha_max=0.7, alpha_scale=0.06,
    beta_start=0, beta_max=0.4,
    gamma_start=0, gamma_max=2.6,  
    log_interval=500, early_stopping_patience=350,
    min_targets=20, min_TFs=1, device=None, return_outputs=True, verbose=True
):

    """
    Train the scRNA_VAE model using in-memory data.
    
    This function trains a variational autoencoder (VAE) designed to infer transcription factor (TF) activities and reconstruct RNA expression. It takes an AnnData object and a gene regulatory network (GRN) prior to initialize the model.
    
    Args:
        rna_data (AnnData, required): AnnData object containing RNA data (normalized expression in `.X` and gene names in `.var.index`).
        net (pd.DataFrame, required): GRN DataFrame with columns:
            - 'source': Transcription factors (TFs).
            - 'target': Target genes.
            - 'weight': Interaction weights between TFs and genes.
        encode_dims (list, optional): List of integers specifying the sizes of hidden layers for the encoder. Default is [512].
        decode_dims (list, optional): List of integers specifying the sizes of hidden layers for the decoder. Default is [1024].
        z_dim (int, optional): Latent space dimensionality. Default is 40.
        train_val_split_ratio (float, optional): Ratio of the training data to the entire dataset. Default is 0.8 (80% training, 20% validation).
        random_state (int, optional): Seed for reproducible splits. Default is 42.
        batch_size (int, optional): Number of samples per training batch. Default is 3,500.
        epochs (int, optional): Total number of epochs for training. Default is 2,000.
        freeze_epochs (int, optional): Number of epochs during which the mask factor is gradually applied to enforce prior structure. Default is 1,500.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 2.5e-4.
        alpha_start (float, optional): Initial value of `alpha`, controlling the weight of initialized TF activities. Default is 0.
        alpha_max (float, optional): Maximum value of `alpha` during training. Default is 0.7.
        alpha_scale (float, optional): Scaling factor for `alpha` during training. Default is 0.06.
        beta_start (float, optional): Initial value of `beta`, controlling the weight of the KL divergence term in the loss. Default is 0.
        beta_max (float, optional): Maximum value of `beta` during training. Default is 0.4.
        gamma_start (float, optional): Initial value of `gamma`, controlling the L1 regularization of TF-target weights. Default is 0.
        gamma_max (float, optional): Maximum value of `gamma` during training. Default is 2.6.
        log_interval (int, optional): Frequency (in epochs) for logging training progress. Default is 500.
        early_stopping_patience (int, optional): Number of epochs to wait for improvement before stopping training early. Default is 350.
        min_targets (int, optional): Minimum number of target genes per TF to retain in the GRN prior. Default is 10.
        min_TFs (int, optional): Minimum number of TFs per target gene to retain in the GRN prior. Default is 1.
        device (str, optional): Device for computation ('cuda' or 'cpu'). If not specified, defaults to 'cuda' if available.
        return_outputs (bool, optional): If True, returns both the trained model and the modified AnnData object with embeddings and reconstructions. Default is True.
        verbose (bool, optional): If True, enable detailed logging. Default is True.
    
    Returns:
        model (scRNA_VAE): Trained scRNA_VAE model instance.
        Optional:
            rna_data (AnnData): Updated AnnData object with the following fields:
                - `obsm["latent"]`: Latent space embedding (cells x latent dimensions).
                - `obsm["TF"]`: Inferred transcription factor activities (cells x TFs) from the layer before the last layer of the model.
                - `obsm["recon_X"]`: Reconstructed RNA expression (cells x genes).
                - `layers["RNA"]`: Original RNA expression data (cells x genes).
                - `uns["GRN_prior"]`: Dictionary containing:
                    - `"matrix"`: The GRN prior weight matrix (TFs x genes).
                    - `"TF_names"`: List of transcription factor names.
                    - `"gene_names"`: List of gene names.
                - `uns["GRN_posterior"]`: Dictionary containing:
                    - `"matrix"`: The GRN posterior weight matrix (TFs x genes).
                    - `"TF_names"`: List of transcription factor names.
                    - `"gene_names"`: List of gene names.
    
    Notes:
        - The TF activities stored in `rna_data.obsm["TF"]` reflect the model's inferred activities, not the raw ULM estimates. 
        - Raw ULM estimates are stored as `rna_data.obsm["ulm_estimate"]`.
        - `rna_data.uns["GRN_prior"]` contains the aligned GRN matrix fed to the model as a prior.
        - `rna_data.uns["GRN_posterior"]` contains the inferred GRN matrix learned by the model.
        - Training halts early if the loss stops improving for `early_stopping_patience` epochs after `alpha` reaches its maximum value.
        - Ensure that `net` and `rna_data` overlap in their genes to avoid errors during initialization.
    
    Examples:
        >>> trained_model, updated_data = train_model(
        >>>     rna_data=my_rna_data,
        >>>     net=my_grn
        >>> )
        >>> GRN_prior = updated_data.uns["GRN_prior"]["matrix"]
        >>> GRN_posterior = updated_data.uns["GRN_posterior"]["matrix"]
    """
    if not (0 < train_val_split_ratio < 1):
        raise ValueError("train_val_split_ratio must be between 0 and 1.")
    if encode_dims is None:
        encode_dims = [512]
    if decode_dims is None:
        decode_dims = [1024]

    set_random_seed(seed=random_state)
    rna_data = rna_data.copy()
    
    # Set logging verbosity, control verbosity of train_model logger only.
    if verbose:
        train_logger.setLevel(logging.INFO)
    else:
        train_logger.setLevel(logging.WARNING)
        
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine batch size dynamically if not provided
    if batch_size is None:
        batch_size = int(train_val_split_ratio * rna_data.n_obs)  
        train_logger.info(f"No batch size provided. Using default batch size equal to the number of training samples: {batch_size}")
    else:
        train_logger.info(f"Using provided batch size: {batch_size}")

    train_logger.info("=" * 40)
    train_logger.info("Starting scRegulate TF inference Training Pipeline")
    train_logger.info("=" * 40)

    # Adapt prior and prepare data
    W_prior, gene_names, TF_names = adapt_prior_and_data(rna_data, net, min_targets, min_TFs)

    # Split into training and validation sets
    train_logger.info(f"Splitting data with train-validation split ratio={train_val_split_ratio}")
    n_cells = rna_data.n_obs
    all_indices = np.arange(n_cells)
    train_indices = np.random.choice(all_indices, size=int(train_val_split_ratio * n_cells), replace=False)
    val_indices = np.setdiff1d(all_indices, train_indices)
    
    # Use the indices to reorder cell names
    train_cell_names = rna_data.obs.index[train_indices]
    val_cell_names = rna_data.obs.index[val_indices]
    shuffled_cell_names = np.concatenate([train_cell_names, val_cell_names])
    
    # Align metadata to the reordering applied to the data
    rna_data = rna_data.copy()
    rna_data = rna_data[shuffled_cell_names, :]  # This reorders both .obs and .X
    
    # Validation step: Ensure alignment
    assert np.array_equal(rna_data.obs.index, shuffled_cell_names), "Mismatch in obs index and shuffled cell names"

    # Run ULM once for the entire dataset
    train_logger.info("Running ULM...")
    ulm_start = time.time()
    ulm.run_ulm(rna_data, net=net, min_n=min_targets, batch_size=batch_size, source="source", target="target", weight="weight", verbose=verbose)
    train_logger.info(f"ULM completed in {time.time() - ulm_start:.2f}s")
    ulm_estimates = torch.tensor(rna_data.obsm["ulm_estimate"].loc[:, TF_names].to_numpy(), dtype=torch.float32)

    # Split ULM estimates
    ulm_estimates_train = ulm_estimates[train_indices, :]
    ulm_estimates_val = ulm_estimates[val_indices, :]

    # Prepare data for training and validation
    gene_indices = [rna_data.var.index.get_loc(gene) for gene in gene_names]
    scRNA_train = torch.tensor(
        rna_data.X[train_indices][:, gene_indices].todense() if not isinstance(rna_data.X, np.ndarray) else rna_data.X[train_indices][:, gene_indices],
        dtype=torch.float32
    )
    scRNA_val = torch.tensor(
        rna_data.X[val_indices][:, gene_indices].todense() if not isinstance(rna_data.X, np.ndarray) else rna_data.X[val_indices][:, gene_indices],
        dtype=torch.float32
    )
    
    # Validate ULM estimates
    train_logger.info("Validating ULM estimates...")
    # Validate ULM estimates
    if ulm_estimates_train.isnan().any() or ulm_estimates_val.isnan().any():
        train_logger.error("ULM estimates contain NaN values. Check data preprocessing.")
        raise ValueError("Invalid ULM estimates.")
    train_logger.info(f"ULM estimates validation passed. Shape: {ulm_estimates.shape}")

    # Transfer data to device
    train_logger.info(f"Transferring data to device {device}")
    scRNA_train, scRNA_val = scRNA_train.to(device), scRNA_val.to(device)
    ulm_estimates_train, ulm_estimates_val = ulm_estimates_train.to(device), ulm_estimates_val.to(device)
    W_prior = W_prior.to(device)
    train_logger.info("Transfer complete...")
    
    # Clear memory after transfer
    clear_memory(device)
    
    # Initialize model
    model = scRNA_VAE(scRNA_train.shape[1], encode_dims, decode_dims, z_dim, W_prior.shape[1]).to(device)
    model.W_prior = W_prior
    model.gene_names = gene_names

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=200, factor=0.5, min_lr=1e-6)
    mask = (W_prior == 0).float()
    
    best_val_loss, epochs_no_improve = float("inf"), 0
    batch_size = int(batch_size)
    train_loader = create_dataloader(scRNA_train, batch_size=batch_size)
    val_loader = create_dataloader(scRNA_val, batch_size=batch_size)
    start_time = time.time()

    alpha, beta, gamma = alpha_start, beta_start, gamma_start  # Initialize parameters

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss, total_samples = 0, 0
        mask_factor = schedule_mask_factor(epoch, freeze_epochs)

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch.size(0)
            tf_activity_slice = ulm_estimates_train[start_idx:end_idx]
            if tf_activity_slice.size(0) != batch.size(0):
                continue

            # Forward pass and loss calculation
            recon_batch, mu, logvar = model(batch, tf_activity_init=tf_activity_slice, alpha=alpha)
            MSE, KLD = loss_function(recon_batch, batch, mu, logvar)
            loss = MSE + beta * KLD + gamma * torch.sum(torch.abs(model.tf_mapping.weight) * mask)
            loss.backward()

            # Apply gradient masking and clipping
            if epoch < freeze_epochs:
                apply_gradient_mask(model.tf_mapping, mask, mask_factor)
            clip_gradients(model)
            optimizer.step()

            total_loss += loss.detach() #* len(batch)
            total_samples += len(batch)

        avg_train_loss = compute_average_loss(total_loss, total_samples)

        # Evaluate on validation set
        model.eval()
        val_loss, val_samples = 0, 0
        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(val_loader):
                start_idx = val_batch_idx * batch_size
                end_idx = start_idx + val_batch.size(0)
                tf_val_slice = ulm_estimates_val[start_idx:end_idx]
                if tf_val_slice.size(0) != val_batch.size(0):
                    continue

                recon_val, mu_val, logvar_val = model(val_batch, tf_activity_init=tf_val_slice, alpha=alpha)
                MSE_val, KLD_val = loss_function(recon_val, val_batch, mu_val, logvar_val)
                loss_val = MSE_val + beta * KLD_val + gamma * torch.sum(torch.abs(model.tf_mapping.weight) * mask)
                val_loss += loss_val.detach() #* len(val_batch)
                val_samples += len(val_batch)

        avg_val_loss = compute_average_loss(val_loss, val_samples)

        # Update parameters based on validation loss improvement
        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss  # Update the best validation loss
            alpha = alpha + (alpha_max - alpha_start) * alpha_scale
            beta = schedule_parameter(epoch, beta_start, beta_max, epochs)
            gamma = schedule_parameter(epoch, gamma_start, gamma_max, epochs)



        alpha = min(alpha, alpha_max)
        
        scheduler.step(avg_val_loss)

        # Clear memory after each epoch
        clear_memory(device)

        # Check early stopping
        if early_stopping_patience is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            elif alpha == 1.0:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                train_logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Log progress
        if epoch % log_interval == 0:
            n_features = scRNA_train.shape[1]
            train_loss_per_sample = avg_train_loss / (len(train_indices) * n_features)
            val_loss_per_sample = avg_val_loss / (len(val_indices)* n_features)
            train_logger.info(
                f"Epoch {epoch+1}: Avg Train Loss = {train_loss_per_sample:.4f}, Avg Val Loss = {val_loss_per_sample:.4f}, "
                f"Alpha = {alpha:.4f}, Beta = {beta:.4f}, Gamma = {gamma:.4f}, Mask Factor: {mask_factor:.2f}"
            )

    # End timing and log final metrics
    total_time = time.time() - start_time
    train_logger.info("Training completed in %.2fs", total_time)

    clear_memory(device)
    # Add latent space, TF activities, reconstructed RNA, and original RNA to AnnData
    if return_outputs:
        latent_space = []
        TF_space = []
        reconstructed_rna = []
        model.eval()
    
        # Use the model's batch encoding to extract relevant spaces
        # Encode all samples (train + val) together
        # Encode all samples (train + val) together
        latent_space = np.concatenate([
            model.encodeBatch(loader, device=device, out='z') for loader in [train_loader, val_loader]
        ], axis=0)
        # Reorder latent embeddings to match RNA order
        latent_space = latent_space[np.argsort(np.concatenate([train_indices, val_indices])), :]

        
        TF_space = np.concatenate([
            model.encodeBatch(loader, device=device, out='tf') for loader in [train_loader, val_loader]
        ], axis=0)

        # Reorder TF embeddings to match RNA order
        TF_space = TF_space[np.argsort(np.concatenate([train_indices, val_indices])), :]

        
        reconstructed_rna = np.concatenate([
            model.encodeBatch(loader, device=device, out='x') for loader in [train_loader, val_loader]
        ], axis=0)

        # Reorder reconstructed_rna embeddings to match RNA order
        reconstructed_rna = reconstructed_rna[np.argsort(np.concatenate([train_indices, val_indices])), :]

        
        # Extract posterior weight matrix
        W_posterior = model.tf_mapping.weight.detach().cpu().numpy()

        # Reorder obs based on concatenated train and validation indices
        meta_obs = rna_data.obs.copy()

        rna_modality = sc.AnnData(
            X=rna_data.X,  # Original RNA expression
            obs=meta_obs,
            var=rna_data.var.copy(),
            obsm=rna_data.obsm.copy(),
            uns=rna_data.uns.copy(),
            layers=rna_data.layers.copy() if hasattr(rna_data, "layers") else None
        )

        rna_modality.uns["type"] = "RNA"
        
        tf_modality = sc.AnnData(
            X=TF_space,
            obs=meta_obs,
            var=pd.DataFrame(index=TF_names)  # Index of transcription factors
        )
        tf_modality.uns["type"] = "TF"
        
        recon_rna_modality = sc.AnnData(
            X=reconstructed_rna,
            obs=meta_obs,
            var=pd.DataFrame(index=gene_names)  # Use aligned gene names
        )
        recon_rna_modality.uns["type"] = "Reconstructed RNA"
        
        latent_modality = sc.AnnData(
            X=latent_space,
            obs=meta_obs,
            var=pd.DataFrame(index=[f"latent_{i}" for i in range(latent_space.shape[1])])
        )
        latent_modality.uns["type"] = "Latent Space"
        
        # Add modalities to the parent AnnData object
        rna_data.modality = {
            "RNA": rna_modality,
            "TF": tf_modality,
            "recon_RNA": recon_rna_modality,
            "latent": latent_modality
        }

        rna_data.uns["main_modality"] = "RNA"  # Set default modality
        rna_data.uns["current_modality"] = "RNA"  # Track current modality
        train_logger.info(f"Default modality set to: {rna_data.uns['main_modality']}")
        train_logger.info(f"Current modality: {rna_data.uns['current_modality']}")

        
        # Store GRN_prior in .uns
        rna_data.uns["GRN_prior"] = {
            "matrix": W_prior.cpu().numpy(),
            "TF_names": TF_names,
            "gene_names": gene_names,
        }

        # Store normalized GRN_posterior in .uns
        rna_data.uns["GRN_posterior"] = {
            "matrix": MinMaxScaler().fit_transform(model.tf_mapping.weight.detach().cpu().numpy()),
            "TF_names": TF_names,
            "gene_names": gene_names,
        }
  
        GRN_posterior = pd.DataFrame(
            rna_data.uns['GRN_posterior']['matrix'].copy(),
            index=rna_data.uns['GRN_posterior']['gene_names'],
            columns=rna_data.uns['GRN_posterior']['TF_names']
        )
        
        train_logger.info("`GRN_prior` and `GRN_posterior` stored in the AnnData object under .uns")

        # Final logging
        train_logger.info("=" * 40)
        train_logger.info("[FINAL SUMMARY]")
        train_logger.info(f"Training stopped after {epoch + 1} epochs.")
        train_logger.info(f"Final Train Loss: {avg_train_loss:.4f}")
        train_logger.info(f"Final Valid Loss: {avg_val_loss:.4f}")
        train_logger.info(f"Final Alpha: {alpha:.4f}, Beta: {beta:.4f}, Gamma: {gamma:.4f}")
        train_logger.info(f"Total Training Time: {total_time:.2f}s")
        
        # Retrieve and log shapes from modalities
        train_logger.info(f"Latent Space Shape: {rna_data.modality['latent'].X.shape}")
        train_logger.info(f"TF Space Shape: {rna_data.modality['TF'].X.shape}")
        train_logger.info(f"Reconstructed RNA Shape: {rna_data.modality['recon_RNA'].X.shape}")
        train_logger.info(f"Original RNA Shape: {rna_data.modality['RNA'].X.shape}")
        
        train_logger.info(f"TFs: {len(TF_names)}, Genes: {rna_data.modality['RNA'].X.shape[1]}")
        train_logger.info("=" * 40)

    
        return model, rna_data, GRN_posterior
    
    return model
