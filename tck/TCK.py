from concurrent.futures import ProcessPoolExecutor
import numpy as np
import tqdm
from sklearn.preprocessing import normalize
from .GMM_MAP_EM import GMM_MAP_EM
from .GMMposterior import GMMposterior

class TCK:
    def __init__(self, G=30, C=None):
        self.G = G
        self.C = C


    def process_gmm(self, X, G, minN, minT, maxT, minV, maxV, I, missing, i):
        """
        Function to parallelize the training of the TCK.
        """
        c = (i // G) + 2
        kwargs = {'C': c, 'minN': minN, 'minT': minT, 'maxT': maxT, 'minV': minV, 'maxV': maxV, 'I': I, 'missing': missing}
        return GMM_MAP_EM(X, **kwargs)
    

    def update_matrix(self, i, mode, Xte):
        """
        Function to parallelize the training of the TCK.
        """
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
        """
        Fit the TCK model to the training data.

        Parameters:
        X (numpy.ndarray): Training data of size N x T x V.
        minN (float): Minimum percentage of samples to be used in the training of the GMMs.
        minV (int): Minimum number of attributes to be sampled from the dataset.
        maxV (int): Maximum number of attributes to be sampled from the dataset.
        minT (int): Minimum length of time segments to be sampled from the dataset.
        maxT (int): Maximum length of time segments to be sampled from the dataset.
        I (int): Number of iterations for the MAP-EM algorithm.

        Returns:
        TCK: The fitted TCK model.
        """
        
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
      
        return self

    
    def predict(self, mode, Xte=None):
        """
        Predict the similarity matrix for the test data.

        Parameters:
        mode (str): The mode of the prediction. Possible values are 'tr-tr', 'tr-te', and 'te-te'.
        Xte (numpy.ndarray): The test data of size N x T x V.

        Returns:
        numpy.ndarray: The similarity matrix.
        """

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