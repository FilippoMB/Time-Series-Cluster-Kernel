import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
from sklearn.metrics import v_measure_score
from tck.TCK import TCK
from tck.datasets import DataLoader

np.random.seed(0)

#%% ============ Load dataset =======================
downloader = DataLoader()
Xtr, Ytr, Xte, Yte = downloader.get_data('ECG2D') 

# Since we are doing clustering, we do not need the train/test split
X = np.concatenate((Xtr, Xte), axis=0)
Y = np.concatenate((Ytr, Yte), axis=0)

#%% ============ Missing data (40%) =================
mask = np.random.choice([0, 1], size=X.shape, p=[0.6, 0.4])
X[mask == 1] = np.nan

#%% ============ Compute TCK ========================
tck = TCK(G=30, C=15)
K = tck.fit(X).predict(mode='tr-tr')
print(f"K shape: {K.shape}")

#%% ============ Perform clustering =============

# Compute Dissimilarity matrix
Dist = 1.0 - (K+K.T)/(2.0*K.max())
np.fill_diagonal(Dist, 0) # due to numerical errors, the diagonal might not be 0

# Hierarchical clustering
distArray = ssd.squareform(Dist)
Z = linkage(distArray, 'ward')
clust = fcluster(Z, t=2.5, criterion="distance")
print(f"Found {len(np.unique(clust))} clusters")

# Evaluate the agreement between class and cluster labels
nmi = v_measure_score(Y[:,0], clust)
print(f"Normalized Mutual Information (v-score): {nmi:.3f}")
# %%
