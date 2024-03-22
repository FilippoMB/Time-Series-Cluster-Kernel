from setuptools import setup, find_packages

setup(
    name="tck",
    version="0.1",
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>1.19.5',
        'scikit_learn>=1.4',
        'scipy',
        'requests',
        'tqdm'
    ],
    author="Filippo Maria Bianchi",
    author_email="filippombianchi@gmail.com",
    description="Kernel similarity for classification and clustering of multi-variate time series with missing values.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/FilippoMB/Time-Series-Cluster-Kernel",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)