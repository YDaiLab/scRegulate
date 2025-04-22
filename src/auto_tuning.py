import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler, NSGAIISampler, QMCSampler
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner

import logging
from .utils import set_active_modality

from .fine_tuning import fine_tune_clusters
from .train import train_model
from .train import adapt_prior_and_data

# ---------- Logger Configuration ----------
autotune_logger = logging.getLogger("autotune")
autotune_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s - %(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
if not autotune_logger.handlers:
    autotune_logger.addHandler(handler)
autotune_logger.propagate = False

# ---------- Common Validation Loss Function ----------
def compute_validation_loss(model, val_adata):
    autotune_logger.info("Computing validation loss...")

    # Retrieve gene names used in the model
    if not hasattr(model, "gene_names"):
        raise ValueError("Model does not have gene_names attribute. Ensure it is set during training.")
    model_genes = model.gene_names  # List of gene names used during training

    # Align validation data with model's gene names
    adata_genes = val_adata.var_names
    shared_genes = set(adata_genes).intersection(set(model_genes))
    if len(shared_genes) < len(model_genes):
        val_adata = val_adata[:, list(shared_genes)].copy()
        autotune_logger.info(f"Aligned validation data to {len(shared_genes)} shared genes.")

    X_val = val_adata.X
    if not isinstance(X_val, np.ndarray):
        X_val = X_val.toarray()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(X_val_tensor)
        z = model.reparameterize(mu, logvar)
        tf_activity = model.decode(z)
        recon_x = model.tf_mapping(tf_activity)
        mse_loss = torch.mean((recon_x - X_val_tensor) ** 2).item()

    autotune_logger.info(f"Validation MSE loss: {mse_loss}")
    return mse_loss

# ---------- Training Objective Function ----------
def training_objective(trial, train_adata, val_adata, net, train_model, **kwargs):
    autotune_logger.info("Running training objective...")

    # Suggest hyperparameters
    freeze_epochs = trial.suggest_int('freeze_epochs', kwargs['freeze_epochs_range'][0], kwargs['freeze_epochs_range'][1])
    alpha_scale = trial.suggest_float('alpha_scale', kwargs['alpha_scale_range'][0], kwargs['alpha_scale_range'][1])
    alpha_max = trial.suggest_float('alpha_max', kwargs['alpha_max_range'][0], kwargs['alpha_max_range'][1])
    beta_max = trial.suggest_float('beta_max', kwargs['beta_max_range'][0], kwargs['beta_max_range'][1])
    gamma_max = trial.suggest_float('gamma_max', kwargs['gamma_max_range'][0], kwargs['gamma_max_range'][1])
    learning_rate = trial.suggest_float('learning_rate', kwargs['learning_rate_range'][0], kwargs['learning_rate_range'][1], log=True)

    # Train model
    model, _, _ = train_model(
        rna_data=train_adata,
        net=net,
        encode_dims=kwargs['encode_dims'],
        decode_dims=kwargs['decode_dims'],
        z_dim=kwargs['z_dim'],
        epochs=kwargs['epochs'],
        freeze_epochs=freeze_epochs,
        learning_rate=learning_rate,
        
        alpha_start=0,
        alpha_scale=alpha_scale,
        alpha_max=alpha_max,
        
        beta_start=0,
        beta_max=beta_max,
        
        gamma_start=0,
        gamma_max=gamma_max,
        
        log_interval=kwargs['log_interval'],
        early_stopping_patience=kwargs['early_stopping_patience'],
        min_targets=kwargs['min_targets'],
        min_TFs=kwargs['min_TFs'],
        device=None,
        return_outputs=True,
        verbose=kwargs['verbose']
    )

    # Compute validation loss
    val_loss = compute_validation_loss(model, val_adata)
    return val_loss
    
# ---------- Fine-Tuning Objective Function ----------
def fine_tuning_objective(trial, processed_adata, model, **kwargs):
    autotune_logger.info("Running fine-tuning objective...")

    # Suggest hyperparameters
    beta_max = trial.suggest_float("beta_max", kwargs['beta_max_range'][0], kwargs['beta_max_range'][1])
    max_weight_norm = trial.suggest_float("max_weight_norm", kwargs['max_weight_norm_range'][0], kwargs['max_weight_norm_range'][1])
    tf_mapping_lr = trial.suggest_float("tf_mapping_lr", kwargs['tf_mapping_lr_range'][0], kwargs['tf_mapping_lr_range'][1], log=True)
    fc_output_lr = trial.suggest_float("fc_output_lr", kwargs['fc_output_lr_range'][0], kwargs['fc_output_lr_range'][1], log=True)
    default_lr = trial.suggest_float("default_lr", kwargs['default_lr_range'][0], kwargs['default_lr_range'][1], log=True)    

    # Log modality availability
    autotune_logger.info("Checking modalities in processed_adata...")
    if not hasattr(processed_adata, "modality"):
        autotune_logger.error("processed_adata does not have 'modality' attribute.")
        raise ValueError("Missing 'modality' attribute in processed_adata.")

    if "TF" not in processed_adata.modality:
        autotune_logger.error(f"Available modalities: {list(processed_adata.modality.keys())}")
        raise ValueError("Missing modality['TF'] in processed_adata.")

    autotune_logger.info(f"Modalities available: {list(processed_adata.modality.keys())}")

    # Log cluster_key and its addition to obs
    cluster_key = kwargs.get('cluster_key', None)
    autotune_logger.info(f"Adding cluster_key '{cluster_key}' to processed_adata.obs...")

    # Fine-tune model
    _, _, fine_model , _ = fine_tune_clusters(
        processed_adata=processed_adata,
        model=model,
        cluster_key=cluster_key,
        epochs=kwargs['epochs'],
        device=kwargs['device'],
        verbose=kwargs['verbose'],
        beta_start=0,
        beta_max=beta_max,
        max_weight_norm=max_weight_norm,
        early_stopping_patience=kwargs['early_stopping_patience'],
        tf_mapping_lr=tf_mapping_lr,
        fc_output_lr=fc_output_lr,
        default_lr=default_lr  # Fine-tuned base learning rate
    )

    # Compute validation loss
    autotune_logger.info("Computing validation loss after fine-tuning...")
    val_loss = compute_validation_loss(fine_model, processed_adata)
    return val_loss


# ---------- Unified Auto-Tuning Function ----------
def auto_tune(
    mode,  # 'training' or 'fine-tuning'
    processed_adata=None,
    model=None,
    net=None,
    train_model=None,  # Explicitly added as a direct argument
    n_trials=10,
    **kwargs
):
    autotune_logger.info(f"Starting auto-tuning for {mode} with {n_trials} trials...")

    if processed_adata is None:
        raise ValueError("processed_adata must be provided for auto-tuning.")

    
    if mode == "training":
        train_val_split_ratio = kwargs.get('train_val_split_ratio', 0.8)
        np.random.seed(0)
        indices = np.arange(processed_adata.n_obs)
        np.random.shuffle(indices)
        train_size = int(train_val_split_ratio * len(indices))
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

        train_adata = processed_adata[train_idx].copy()
        val_adata = processed_adata[val_idx].copy()

        # Adapt prior for training data
        W_prior, gene_names, TF_names = adapt_prior_and_data(train_adata, net, kwargs['min_targets'], kwargs['min_TFs'])

        # Align validation data to training gene names
        val_genes = val_adata.var_names.intersection(gene_names)
        val_adata = val_adata[:, list(val_genes)].copy()

    elif mode == "fine-tuning":
        # Ensure the active modality is set to 'RNA'
        autotune_logger.info("Setting the active modality to 'RNA'...")
        processed_adata = set_active_modality(processed_adata, "RNA")
        autotune_logger.info("Active modality set to 'RNA'.")

        if not hasattr(model, "gene_names"):
            raise ValueError("Model does not have gene_names attribute. Ensure it is set during training.")
        gene_names = model.gene_names
        fine_tuned_genes = processed_adata.var_names.intersection(gene_names)
        processed_adata._inplace_subset_var(list(fine_tuned_genes))
        autotune_logger.info(f"Aligned processed_adata to {len(fine_tuned_genes)} genes.")
        train_adata, val_adata = None, None
    else:
        raise ValueError("Invalid mode. Choose 'training' or 'fine-tuning'.")

    # Create study
    #study = optuna.create_study(direction='minimize', sampler=TPESampler())

    # Use BOHB (Bayesian Optimization with HyperBand)
    sampler = TPESampler(multivariate=True, seed=42)

    pruner = HyperbandPruner(
        min_resource=500,         # Let model run at least 500 epochs before considering pruning
        max_resource=kwargs['epochs'],        # Full training budget
        reduction_factor=3,       # Prune bottom 2/3 at each round
         bootstrap_count=0 
    )
    
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    
    if mode == "training":
        study.optimize(
            lambda trial: training_objective(
                trial, train_adata, val_adata, net, train_model, **kwargs
            ),
            n_trials=n_trials
        )
    elif mode == "fine-tuning":
        study.optimize(
            lambda trial: fine_tuning_objective(
                trial, processed_adata, model, **kwargs
            ),
            n_trials=n_trials
        )
    else:
        raise ValueError("Invalid mode. Choose 'training' or 'fine-tuning'.")

    autotune_logger.info(f"Best hyperparameters: {study.best_params}")
    autotune_logger.info(f"Best validation loss: {study.best_value}")
    return study.best_params, study.best_value
