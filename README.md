[![arXiv](https://img.shields.io/badge/arXiv-1803.07870-b31b1b.svg)](https://arxiv.org/abs/1704.00794)

The Time Series Cluster Kernel (TCK) is a kernel similarity for multi-variate time series with missing values. The kernel can be used to perform classification, clustering, and dimensionality reduction tasks.

<img src="docs/tck_scheme.png" style="width: 18cm" align="center">

TCK is based on an ensemble of Gaussian Mixture Models for time series that use informative Bayesian priors robut sto missing values.


# Installation

The recommended installation is with pip:

````bash
pip install tck
````

Alternatively, you can install the library from source:
````bash
git clone https://github.com/FilippoMB/https://github.com/FilippoMB/Time-Series-Cluster-Kernel.git
cd https://github.com/FilippoMB/Time-Series-Cluster-Kernel
pip install -e .
````

# Quick start

The following scripts provide minimalistic examples that illustrate how to use the library for different tasks.

To run them, download the project and cd to the root folder:

````bash
git clone https://github.com/FilippoMB/https://github.com/FilippoMB/Time-Series-Cluster-Kernel.git
cd https://github.com/FilippoMB/Time-Series-Cluster-Kernel
````

**Classification**

````bash
python examples/classification.py
````

**Clustering**

````bash
python examples/clustering.py
````

The following notebooks illustrate more advanced use-cases.

- Perform time series cluster analysis and visualize the results: [view]() or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

## Running on Windows

TCK uses multiprocessing. While using multiprocessing in Python on windows, it is necessary to protect the entry point of the program by using 

```python
if __name__ == '__main__':
```

Please, refer to the following examples.

**Classification**

````bash
python examples/classification_windows.py
````

**Clustering**

````bash
python examples/clustering_windows.py
````

## Configuration


# Citation

Please, consider citing the original paper if you are using this library in your reasearch

````bibtex
@article{mikalsen2018time,
  title={Time series cluster kernel for learning similarities between multivariate time series with missing data},
  author={Mikalsen, Karl {\O}yvind and Bianchi, Filippo Maria and Soguero-Ruiz, Cristina and Jenssen, Robert},
  journal={Pattern Recognition},
  volume={76},
  pages={569--581},
  year={2018},
  publisher={Elsevier}
}
````