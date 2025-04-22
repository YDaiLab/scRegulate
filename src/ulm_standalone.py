"""
Standalone Implementation of the Univariate Linear Model (ULM).

This implementation is based on the DecoupleR package's `run_ulm` function but 
has been adapted to work independently without any external dependencies. It 
calculates transcription factor activity scores from a gene expression matrix 
and a regulatory network.

Citation:
- For the original ULM method, please cite DecoupleR:
  Badia-i-Mompel, P., Wessels, L., & Reinders, M. (2021). DecoupleR: a computational framework 
  to infer molecular activities from omics data. *Bioinformatics*.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import t
from tqdm.auto import tqdm
import logging
import warnings

logger = logging.getLogger(__name__)
from .utils import set_random_seed
set_random_seed()
# --- Helper Functions ---

def mat_cov(A, b):
    return np.dot(b.T - b.mean(), A - A.mean(axis=0)) / (b.shape[0] - 1)

def mat_cor(A, b):
    cov = mat_cov(A, b)
    ssd = np.std(A, axis=0, ddof=1) * np.std(b, axis=0, ddof=1).reshape(-1, 1)
    return cov / ssd

def t_val(r, df):
    return r * np.sqrt(df / ((1.0 - r + 1.0e-16) * (1.0 + r + 1.0e-16)))

def filt_min_n(c, net, min_n=5):
    msk = np.isin(net['target'].values.astype('U'), c)
    net = net.iloc[msk]

    sources, counts = np.unique(net['source'].values.astype('U'), return_counts=True)
    msk = np.isin(net['source'].values.astype('U'), sources[counts >= min_n])

    net = net[msk]
    if net.shape[0] == 0:
        raise ValueError(f"No sources with more than min_n={min_n} targets.")
    return net

def match(c, r, net):
    regX = np.zeros((c.shape[0], net.shape[1]), dtype=np.float32)
    c_dict = {gene: i for i, gene in enumerate(c)}
    idxs = [c_dict[gene] for gene in r if gene in c_dict]
    regX[idxs, :] = net[: len(idxs), :]
    return regX

def rename_net(net, source="source", target="target", weight="weight"):
    assert source in net.columns, f"Column '{source}' not found in net."
    assert target in net.columns, f"Column '{target}' not found in net."
    if weight is not None:
        assert weight in net.columns, f"Column '{weight}' not found in net."

    net = net.rename(columns={source: "source", target: "target", weight: "weight"})
    net = net.reindex(columns=["source", "target", "weight"])

    if net.duplicated(["source", "target"]).sum() > 0:
        raise ValueError("net contains repeated edges.")
    return net

def get_net_mat(net):
    X = net.pivot(columns="source", index="target", values="weight")
    X[np.isnan(X)] = 0
    sources = X.columns.values
    targets = X.index.values
    X = X.values
    return sources.astype("U"), targets.astype("U"), X.astype(np.float32)


# --- Main Functionality ---

def ulm(mat, net, batch_size):
    logger.debug("Starting ULM calculation...")
    n_samples = mat.shape[0]
    n_features, n_fsets = net.shape
    df = n_features - 2  # Degrees of freedom

    if isinstance(mat, csr_matrix):
        logger.debug("Matrix is sparse, processing in batches...")
        n_batches = int(np.ceil(n_samples / batch_size))
        es = np.zeros((n_samples, n_fsets), dtype=np.float32)
        # The progress bar is only shown if logger level <= INFO
        show_progress = logger.getEffectiveLevel() <= logging.INFO
        for i in tqdm(range(n_batches), disable=not show_progress):
            start, end = i * batch_size, i * batch_size + batch_size
            batch = mat[start:end].toarray().T
            r = mat_cor(net, batch)
            es[start:end] = t_val(r, df)
    else:
        logger.debug("Matrix is dense, processing at once...")
        r = mat_cor(net, mat.T)
        es = t_val(r, df)

    pv = t.sf(abs(es), df) * 2
    logger.debug("ULM calculation complete.")
    return es, pv


def run_ulm(adata, net, batch_size, source="source", target="target", weight="weight", min_n=5, verbose=True):
    """
    Run ULM on a Scanpy AnnData object and store results in `.obsm`.

    Args:
        adata: AnnData object with gene expression data.
        net: DataFrame with columns [source, target, weight].
        source (str): Name of the TF column in net.
        target (str): Name of the target gene column in net.
        weight (str): Name of the weight column in net.
        batch_size (int): Batch size for ULM if data is large.
        min_n (int): Minimum number of targets per TF.
        verbose (bool): If True, set logging level to INFO, else WARNING.

    Returns:
        adata with ULM results in adata.obsm["ulm_estimate"] and adata.obsm["ulm_pvals"].
    """
    # Set logger level based on verbose
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    logger.info("Initializing ULM analysis...")
    logger.info("Extracting gene expression data...")
    mat = adata.to_df()
    genes = adata.var_names

    logger.info("Preparing the regulatory network...")
    net = rename_net(net, source=source, target=target, weight=weight)
    net = filt_min_n(genes, net, min_n=min_n)

    logger.info("Matching genes in the network to gene expression data...")
    sources, targets, net_mat = get_net_mat(net)
    net_mat = match(genes, targets, net_mat)

    logger.info(f"ULM parameters: {mat.shape[0]} samples, {len(genes)} genes, {net_mat.shape[1]} TFs.")
    logger.info("Calculating ULM estimates and p-values...")
    estimate, pvals = ulm(mat.values, net_mat, batch_size=batch_size)

    logger.info("Processing ULM results...")
    ulm_estimate = pd.DataFrame(estimate, index=mat.index, columns=sources)
    ulm_pvals = pd.DataFrame(pvals, index=mat.index, columns=sources)

    logger.info("Storing ULM results in AnnData object...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # Suppress UserWarnings
        adata.obsm["ulm_estimate"] = np.maximum(ulm_estimate, 0)
        adata.obsm["ulm_pvals"] = ulm_pvals
    
    #adata.obsm["ulm_estimate"] = np.maximum(ulm_estimate, 0)
    #adata.obsm["ulm_pvals"] = ulm_pvals

    logger.info("ULM analysis complete.")
    return adata
