from setuptools import setup, find_packages

setup(
    name='scRegulate',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'scanpy',
        'numpy',
        'pandas',
        'optuna',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'scipy',
    ],
    author='Mehrdad Zandigohar',
    description='scRegulate: Single-cell TF activity inference via regulatory-embedded VAE',
    python_requires='>=3.8',
)
