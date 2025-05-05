from setuptools import setup, find_packages

setup(
    name="scRegulate",
    version="0.1.0",
    author="Mehrdad Zandigohar",
    author_email="mehr.zgohar@gmail.com",
    description="Python Toolkit for Transcription Factor Activity Inference and Clustering of scRNA-seq Data",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YDaiLab/scRegulate",
    project_urls={
        "Documentation": "https://github.com/YDaiLab/scRegulate#readme",
        "Issue Tracker": "https://github.com/YDaiLab/scRegulate/issues",
        "Paper (bioRxiv)": "https://doi.org/10.1101/2025.04.17.649372"
    },
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.23",
        "scanpy>=1.9",
        "anndata>=0.8"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
