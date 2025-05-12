import os
import urllib.request
import pandas as pd

def _download_file_if_needed(filename: str, url: str, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, filename)
    if not os.path.exists(filepath):
        print(f"[scRegulate] Downloading {filename} to {filepath}...")
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}.\nError: {e}")
    return filepath

def collectri_prior(species: str = "human") -> pd.DataFrame:
    """
    Returns the collectri prior dataframe for the given species ("human" or "mouse").
    Downloads the file on first use and caches it under ~/.scregulate/priors/.

    Parameters:
    - species: str = "human" or "mouse"

    Returns:
    - pd.DataFrame with TF-target prior network
    """
    base_url = "https://github.com/YDaiLab/scRegulate/raw/main/priors/"
    species_to_filename = {
        "human": "collectri_human_net.csv",
        "mouse": "collectri_mouse_net.csv"
    }

    if species not in species_to_filename:
        raise ValueError("species must be either 'human' or 'mouse'")

    filename = species_to_filename[species]
    url = base_url + filename
    target_dir = os.path.expanduser("~/.scregulate/priors")

    local_path = _download_file_if_needed(filename, url, target_dir)
    return pd.read_csv(local_path)
