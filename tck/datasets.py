import requests
import numpy as np
from io import BytesIO

class DataLoader:
    def __init__(self):
        self.datasets = {
            'AtrialFibrillation': ('https://zenodo.org/records/10852712/files/AF.npz?download=1', 'Multivariate time series classification.\nSamples: 5008 (4823 training, 185 test)\nFeatures: 2\nClasses: 3\nTime series length: 45'),
            'ArabicDigits': ('https://zenodo.org/records/10852747/files/ARAB.npz?download=1', 'Multivariate time series classification.\nSamples: 8800 (6600 training, 2200 test)\nFeatures: 13\nClasses: 10\nTime series length: 93'),
            'Auslan': ('https://zenodo.org/records/10839959/files/Auslan.npz?download=1', 'Multivariate time series classification.\nSamples: 2565 (1140 training, 1425 test)\nFeatures: 22\nClasses: 95\nTime series length: 136'),
            'CharacterTrajectories': ('https://zenodo.org/records/10852786/files/CHAR.npz?download=1', 'Multivariate time series classification.\nSamples: 2858 (300 training, 2558 test)\nFeatures: 3\nClasses: 20\nTime series length: 205'),
            'CMUsubject16': ('https://zenodo.org/records/10852831/files/CMU.npz?download=1', 'Multivariate time series classification.\nSamples: 58 (29 training, 29 test)\nFeatures: 62\nClasses: 2\nTime series length: 580'),
            'ECG2D': ('https://zenodo.org/records/10839881/files/ECG_2D.npz?download=1', 'Multivariate time series classification.\nSamples: 200 (100 training, 100 test)\nFeatures: 2\nClasses: 2\nTime series length: 152'),
            'Japanese_Vowels': ('https://zenodo.org/records/10837602/files/Japanese_Vowels.npz?download=1', 'Multivariate time series classification.\nSamples: 640 (270 training, 370 test)\nFeatures: 12\nClasses: 9\nTime series length: 29'),
            'KickvsPunch': ('https://zenodo.org/records/10852865/files/KickvsPunch.npz?download=1', 'Multivariate time series classification.\nSamples: 26 (16 training, 10 test)\nFeatures: 62\nClasses: 2\nTime series length: 841'),
            'Libras': ('https://zenodo.org/records/10852531/files/LIB.npz?download=1', 'Multivariate time series classification.\nSamples: 360 (180 training, 180 test)\nFeatures: 2\nClasses: 15\nTime series length: 45'),
            'NetFlow': ('https://zenodo.org/records/10840246/files/NET.npz?download=1', 'Multivariate time series classification.\nSamples: 1337 (803 training, 534 test)\nFeatures: 4\nClasses: 2\nTime series length: 997'),
            'RobotArm': ('https://zenodo.org/records/10852893/files/Robot.npz?download=1', 'Multivariate time series classification.\nSamples: 164 (100 training, 64 test)\nFeatures: 6\nClasses: 5\nTime series length: 15'),
            'UWAVE': ('https://zenodo.org/records/10852667/files/UWAVE.npz?download=1', 'Multivariate time series classification.\nSamples: 628 (200 training, 428 test)\nFeatures: 3\nClasses: 8\nTime series length: 315'),
            'Wafer': ('https://zenodo.org/records/10839966/files/Wafer.npz?download=1', 'Multivariate time series classification.\nSamples: 1194 (298 training, 896 test)\nFeatures: 6\nClasses: 2\nTime series length: 198'),
            'Chlorine': ('https://zenodo.org/records/10840284/files/CHLO.npz?download=1', 'Univariate time series classification.\nSamples: 4307 (467 training, 3840 test)\nFeatures: 1\nClasses: 3\nTime series length: 166'), 
            'Phalanx': ('https://zenodo.org/records/10852613/files/PHAL.npz?download=1', 'Univariate time series classification.\nSamples: 539 (400 training, 139 test)\nFeatures: 1\nClasses: 3\nTime series length: 80'),
            'SwedishLeaf': ('https://zenodo.org/records/10840000/files/SwedishLeaf.npz?download=1', 'Univariate time series classification.\nSamples: 1125 (500 training, 625 test)\nFeatures: 1\nClasses: 15\nTime series length: 128'),
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