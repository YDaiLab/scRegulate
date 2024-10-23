[![GitHub issues](https://img.shields.io/github/issues/zandigohar/REGULOMIC)](https://github.com/zandigohar/REGULOMIC/issues)
![Conda](https://img.shields.io/conda/dn/conda-forge/REGULOMIC)
![PyPI Downloads](https://img.shields.io/pypi/dm/REGULOMIC)
![Documentation Status](https://readthedocs.org/projects/REGULOMIC/badge/?version=latest)

# REGULOMIC
**R**egulatory **E**mbedded **G**enerative **U**nified **L**earning for **O**ptimized **M**odeling and **I**nference of Transcription Factor Activity and **C**lustering

## Introduction

**REGULOMIC** is a powerful tool designed for the inference of transcription factor activity using advanced generative modeling techniques. It leverages a unified learning framework to optimize the modeling of cellular regulatory networks, providing researchers with accurate insights into transcriptional regulation. With its efficient clustering capabilities, **REGULOMIC** facilitates the analysis of complex biological data, making it an essential resource for studies in genomics and molecular biology.

## Requirements

Before installing and running REGULONIC, ensure you have the following libraries installed:

- **PyTorch** (version 2.0 or higher)
- **NumPy** (version 1.23 or higher)
- **Scanpy** (version 1.9 or higher)
- **Anndata** (version 0.8 or higher)

You can install these dependencies using `pip`:

```bash
pip install torch numpy scanpy anndata
```

## Installation

You can install **REGULONIC** via pip for a lightweight installation:

```bash
pip install regulonic
```

Alternatively, if you want the latest, unreleased version, you can install it directly from the source on GitHub:

```bash
pip install git+https://github.com/zandigohar/REGULONIC.git
```

For users who prefer Conda or Mamba for environment management, you can install **REGULONIC** along with extra dependencies using:

```bash
mamba create -n=regulonic conda-forge::regulonic
```

## License

The code in **REGULONIC** is licensed under the [MIT License](https://opensource.org/licenses/MIT), which permits academic and commercial use, modification, and distribution. 

Please note that any third-party dependencies bundled with **REGULONIC** may have their own respective licenses.

## Citation

If you use **REGULONIC** in your research, please cite:

Zandigohar, M., et al., 2024. **REGULONIC: Regulatory Embedded Generative Unified Learning for Optimized Modeling and Inference of Cellular Transcription Factor Activity.** Journal/Conference Name. [DOI link here]

