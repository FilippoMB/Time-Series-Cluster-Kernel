import requests
from io import BytesIO
import numpy as np
import scipy.io
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from TCK import TCK

if __name__ == '__main__':

    # ============ Load dataset =======================
    data_url = 'https://raw.githubusercontent.com/FilippoMB/Time-series-classification-and-clustering-with-Reservoir-Computing/master/dataset/JpVow.mat'
    response = requests.get(data_url)
    response.raise_for_status()
    data = scipy.io.loadmat(BytesIO(response.content))
    Xtr = data['X']  # shape is [N,T,V]
    if len(Xtr.shape) < 3:
        Xtr = np.atleast_3d(Xtr)
    Ytr = data['Y']  # shape is [N,1]
    Xte = data['Xte']
    if len(Xte.shape) < 3:
        Xte = np.atleast_3d(Xte)
    Yte = data['Yte']
    print(f"Loaded data from {data_url}\nData shapes:\n Xtr: {Xtr.shape}\n Ytr: {Ytr.shape}\n Xte: {Xte.shape}\n Yte: {Yte.shape}")

    # ============ Missing data (40%) =================
    mask_tr = np.random.choice([0, 1], size=Xtr.shape, p=[0.6, 0.4])
    Xtr[mask_tr == 1] = np.nan
    mask_te = np.random.choice([0, 1], size=Xte.shape, p=[0.6, 0.4])
    Xte[mask_te == 1] = np.nan

    # ============ Compute TCK ========================
    tck = TCK(G=30, C=15)
    Ktr = tck.fit(Xtr).predict(mode='tr-tr')
    Kte = tck.predict(Xte=Xte, mode='tr-te').T
    print(f"Ktr shape: {Ktr.shape}\nKte shape: {Kte.shape}")

    # ============ Perform classification =============
    clf = SVC(kernel='precomputed')
    clf.fit(Ktr, Ytr.ravel())
    Ypred = clf.predict(Kte)
    acc = accuracy_score(Yte, Ypred)
    print(f" Test accuracy: {acc:.2f}")

    # ============ Perform clustering =================
    