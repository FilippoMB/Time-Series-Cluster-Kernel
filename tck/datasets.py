import requests
import numpy as np
from io import BytesIO

class DataLoader:
    def __init__(self):
        self.datasets = {
            'Japanese_Vowels': ('https://zenodo.org/records/10837602/files/Japanese_Vowels.npz?download=1', 'Multivariate time series classification.\nSamples: 640 (270 training, 370 test)\nFeatures: 12\nClasses: 9\nTime series length: 29'),
            'ECG2D': ('https://zenodo.org/records/10839881/files/ECG_2D.npz?download=1', 'Multivariate time series classification.\nSamples: 200 (100 training, 100 test)\nFeatures: 2\nClasses: 2\nTime series length: 152'),
            'Auslan': ('https://zenodo.org/records/10839959/files/Auslan.npz?download=1', 'Multivariate time series classification.\nSamples: 2565 (1140 training, 1425 test)\nFeatures: 22\nClasses: 95\nTime series length: 136'),
            'Wafer': ('https://zenodo.org/records/10839966/files/Wafer.npz?download=1', 'Multivariate time series classification.\nSamples: 1194 (298 training, 896 test)\nFeatures: 6\nClasses: 2\nTime series length: 198'),
            'NetFlow': ('https://zenodo.org/records/10840246/files/NET.npz?download=1', 'Multivariate time series classification.\nSamples: 1337 (803 training, 534 test)\nFeatures: 4\nClasses: 2\nTime series length: 997'),
            'SwedishLeaf': ('https://zenodo.org/records/10840000/files/SwedishLeaf.npz?download=1', 'Univariate time series classification.\nSamples: 1125 (500 training, 625 test)\nFeatures: 1\nClasses: 15\nTime series length: 128'),
            'Chlorine': ('https://zenodo.org/records/10840284/files/CHLO.npz?download=1', 'Univariate time series classification.\nSamples: 4307 (467 training, 3840 test)\nFeatures: 1\nClasses: 3\nTime series length: 166'), 
        }

    def available_datasets(self, details=False):
        print("Available datasets:\n")
        for alias, (_, description) in self.datasets.items():
            if details:
                print(f"{alias}\n-----------\n{description}\n")
            else:
                print(alias)

    def get_data(self, alias):

        if alias not in self.datasets:
            raise ValueError(f"Dataset {alias} not found.")

        url, _ = self.datasets[alias]
        response = requests.get(url)
        if response.status_code == 200:

            data = np.load(BytesIO(response.content))
            Xtr = data['Xtr']  # shape is [N,T,V]
            if len(Xtr.shape) < 3:
                Xtr = np.atleast_3d(Xtr)
            Ytr = data['Ytr']  # shape is [N,1]
            Xte = data['Xte']
            if len(Xte.shape) < 3:
                Xte = np.atleast_3d(Xte)
            Yte = data['Yte']
            print(f"Loaded {alias} dataset.\nData shapes:\n Xtr: {Xtr.shape}\n Ytr: {Ytr.shape}\n Xte: {Xte.shape}\n Yte: {Yte.shape}")

            return (Xtr, Ytr, Xte, Yte)
        else:
            print(f"Failed to download {alias} dataset.")
            return None

if __name__ == '__main__':
    # Example usage
    downloader = DataLoader()
    downloader.available_datasets()  # Describe available datasets
    Xtr, Ytr, Xte, Yte = downloader.get_data('NetFlow')  # Download dataset and return data