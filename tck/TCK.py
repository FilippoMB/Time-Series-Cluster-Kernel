from concurrent.futures import ProcessPoolExecutor
import numpy as np
from GMM_MAP_EM import GMM_MAP_EM
from GMMposterior import GMMposterior
import tqdm
from sklearn.preprocessing import normalize

class TCK:
    def __init__(self, G=30, C=None):
        self.G = G
        self.C = C

    # Function to parallelize the training of the TCK
    def process_gmm(self, X, G, minN, minT, maxT, minV, maxV, I, missing, i):
        c = (i // G) + 2
        kwargs = {'C': c, 'minN': minN, 'minT': minT, 'maxT': maxT, 'minV': minV, 'maxV': maxV, 'I': I, 'missing': missing}
        return GMM_MAP_EM(X, **kwargs)
    

    def update_matrix(self, i, mode, Xte):
        if mode == 'tr-tr':
            return np.dot(normalize(self.GMM[i][0], axis=1), normalize(self.GMM[i][0], axis=1).T)
        else:
            missing = np.isnan(Xte).any()
            if mode == 'tr-te':
                c = (i // self.G) + 2
                return np.dot(normalize(self.GMM[i][0], axis=1), normalize(GMMposterior(Xte, c, *self.GMM[i][1:], missing), axis=1).T)
            elif mode == 'te-te':
                c = (i // self.G) + 2
                posterior = normalize(GMMposterior(Xte, c, *self.GMM[i][1:], missing), axis=1)
                return np.dot(posterior, posterior.T)
            else:
                raise ValueError('Invalid training mode')

    def fit(self, X, minN=0.8, minV=None, maxV=None, minT=6, maxT=None, I=20):
        
        N, T, V = X.shape  # Unpack the shape of X        
        if self.C is None:
            self.C = 10 if N < 100 else 40
        if minV is None:
            minV = 1 if V == 1 else 2
        if maxV is None:
            maxV = min(int(np.ceil(0.9 * V)), 15)
        if maxT is None:
            maxT = min(int(np.floor(0.8 * T)), 25)

        self.GMM = []
        
        # Check for missing data
        missing = np.isnan(X).any()
        missing_data_msg = "The dataset contains missing data\n" if missing else "The dataset does not contain missing data\n"
        print(missing_data_msg)
        
        # Training message
        training_msg = f"Training the TCK using the following parameters:\n\tC = {self.C}, G = {self.G}\n"
        training_msg += f"\tNumber of MTS for each GMM: {int(np.floor(minN*N))} - {N} ({int(np.floor(minN*100))} - 100 percent)\n"
        training_msg += f"\tNumber of attributes sampled from [{minV}, {maxV}]\n"
        training_msg += f"\tLength of time segments sampled from [{minT}, {maxT}]\n\n"
        print(training_msg)
        
        # Parallel processing
        with ProcessPoolExecutor() as executor:
            # Pass additional arguments required by process_gmm
            futures = [executor.submit(self.process_gmm, X, self.G, minN, minT, maxT, minV, maxV, I, missing, i) for i in range(self.G * (self.C - 1))]
            for future in tqdm.tqdm(futures, desc='Fitting GMMs'):
                self.GMM.append(future.result())
        
        # kwargs = {'C': None, 'minN': minN, 'minT': minT, 'maxT': maxT, 'minV': minV, 'maxV': maxV, 'I': I, 'missing': missing}
        # for i in tqdm.tqdm(range(G * (C - 1))):
        #     c = (i // G) + 2

        #     kwargs['C'] = c
        #     res.append(GMM_MAP_EM(X, **kwargs))
                
        return self

    
    def predict(self, mode, Xte=None):

        if mode == 'tr-tr':
            K = np.zeros((self.GMM[0][0].shape[0], self.GMM[0][0].shape[0]))
        elif mode == 'tr-te':
            K = np.zeros((self.GMM[0][0].shape[0], Xte.shape[0]))
        elif mode == 'te-te':
            K = np.zeros((Xte.shape[0], Xte.shape[0]))

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.update_matrix, i, mode, Xte) for i in range(self.G * (self.C - 1))]
            for future in tqdm.tqdm(futures, desc=f"Computing TCK ({mode})"):
                K += future.result()
        
        return K