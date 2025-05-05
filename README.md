# scRegulate<img src="https://raw.githubusercontent.com/YDaiLab/scRegulate/main/assets/tool_logo.svg" align="right" width="360" />
**S**ingle-**C**ell **Regula**tory-Embedded Variational Inference of **T**ranscription Factor Activity from Gene **E**xpression


[![GitHub issues](https://img.shields.io/github/issues/YDaiLab/scRegulate)](https://github.com/YDaiLab/scRegulate/issues)
![Conda](https://img.shields.io/conda/dn/conda-forge/scRegulate)
[![PyPI - Project](https://img.shields.io/pypi/v/scRegulate)](https://pypi.org/project/scRegulate/)
![Documentation Status](https://readthedocs.org/projects/scRegulate/badge/?version=latest)

## Introduction
**scRegulate** is a powerful tool designed for the inference of transcription factor activity from single cell/nucleus RNA data using advanced generative modeling techniques. It leverages a unified learning framework to optimize the modeling of cellular regulatory networks, providing researchers with accurate insights into transcriptional regulation. With its efficient clustering capabilities, **scRegulate** facilitates the analysis of complex biological data, making it an essential resource for studies in genomics and molecular biology.

<br>
<img src="https://raw.githubusercontent.com/YDaiLab/scRegulate/main/assets/Visual_Abstract.png" align="center" />
<br>

For further information and example tutorials, please check our [documentation](https://readthedocs.org/projects/scRegulate/badge/?version=latest).

If you have any questions or concerns feel free to [open an issue](https://github.com/YDaiLab/scRegulate/issues).

## Requirements
scRegulate is implemented in the PyTorch framework. Running scRegulate on `CUDA` is highly recommended if available.

Before installing and running scRegulate, ensure you have the following libraries installed:

- **PyTorch** (version 2.0 or higher)
- **NumPy** (version 1.23 or higher)
- **Scanpy** (version 1.9 or higher)
- **Anndata** (version 0.8 or higher)

You can install these dependencies using `pip`:

```bash
pip install torch numpy scanpy anndata
```

## Installation

You can install **scRegulate** via pip for a lightweight installation:

```bash
pip install scRegulate
```

Alternatively, if you want the latest, unreleased version, you can install it directly from the source on GitHub:

```bash
pip install git+https://github.com/YDaiLab/scRegulate.git
```

For users who prefer Conda or Mamba for environment management, you can install **scRegulate** along with extra dependencies using:

```bash
mamba create -n=scRegulate conda-forge::scRegulate
```

## License

The code in **scRegulate** is licensed under the [MIT License](https://opensource.org/licenses/MIT), which permits academic and commercial use, modification, and distribution. 

Please note that any third-party dependencies bundled with **scRegulate** may have their own respective licenses.

## Citation

**scRegulate** manuscript is currently under peer review. 

If you use **scRegulate** in your research, please cite:

Mehrdad Zandigohar, Jalees Rehman and Yang Dai (2025). **scRegulate: Single-Cell Regulatory-Embedded Variational Inference of Transcription Factor Activity from Gene Expression**, Bioinformatics Journal (under review). [DOI link here]

📄 Read the preprint on bioRxiv: [10.1101/2025.04.17.649372](https://doi.org/10.1101/2025.04.17.649372)


