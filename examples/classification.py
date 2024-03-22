import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tck.TCK import TCK
from tck.datasets import DataLoader

np.random.seed(0)

#%% ============ Load dataset =======================
downloader = DataLoader()
Xtr, Ytr, Xte, Yte = downloader.get_data('Japanese_Vowels') 

#%% ============ Missing data (40%) =================
mask_tr = np.random.choice([0, 1], size=Xtr.shape, p=[0.6, 0.4])
Xtr[mask_tr == 1] = np.nan
mask_te = np.random.choice([0, 1], size=Xte.shape, p=[0.6, 0.4])
Xte[mask_te == 1] = np.nan

#%% ============ Compute TCK ========================
tck = TCK(G=30, C=15)
Ktr = tck.fit(Xtr).predict(mode='tr-tr')
Kte = tck.predict(Xte=Xte, mode='tr-te').T
print(f"Ktr shape: {Ktr.shape}\nKte shape: {Kte.shape}")

#%% ============ Perform classification =============
clf = SVC(kernel='precomputed')
clf.fit(Ktr, Ytr.ravel())
Ypred = clf.predict(Kte)
acc = accuracy_score(Yte, Ypred)
print(f" Test accuracy: {acc:.2f}")