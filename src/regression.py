import numpy as np


class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement PolynomialRegression from scratch.
        
        The `degree` argument controls the complexity of the function.  For
        example, degree = 2 would specify a hypothesis space of all functions
        of the form:

            f(x) = ax^2 + bx + c

        You should implement the closed form solution of least squares:
            w = (X^T X)^{-1} X^T y
        
        Do not import or use these packages: scipy, sklearn, sys, importlib.
        Do not use these numpy or internal functions: polynomial, polyfit, polyval, getattr, globals

        Args:
            degree (int): Degree used to fit the data.
        """
        self.degree = degree
        # raise NotImplementedError
    
    def fit(self, features, targets):
        """
        Fit to the given data.

        Hints:
          - Remember to use `self.degree`
          - Remember to include an intercept (a column of all 1s) before you
            compute the least squares solution.

        Args:
            features (np.ndarray): an array of shape [N, 1] containing real-valued inputs.
            targets (np.ndarray): an array of shape [N, 1] containing real-valued targets.
        Returns:
            None (saves model internally)
        """
        self.X = np.ones((features.shape[0],self.degree+1))

        for i in range(features.shape[0]):
            for j in range(self.degree+1):
                self.X[i,j] = features[i]**j

        # print(f"X.shape = {self.X.shape}")
        # print(f"X = {self.X}")

        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X)), self.X.T) ,targets)
        # print(f"w.shape = {self.w.shape}")
        # print(f"w = {self.w}")

        # raise NotImplementedError

    def predict(self, features):
        """
        Given features, use the trained model to predict target estimates. Call
        this after calling fit.

        Args:
            features (np.ndarray): array of shape [N, 1] containing real-valued inputs.
        Returns:
            predictions (np.ndarray): array of shape [N, 1] containing real-valued predictions
        """
        predictions = np.ones((features.shape[0],1))
        # sum = 0.0

        x = np.ones((features.shape[0],self.degree+1))

        for i in range(0,features.shape[0]):
            for j in range(0,self.degree+1):
                x[i,j] = features[i]**j


        for k in range(0,features.shape[0]):
            predictions[k] = np.matmul(self.w.T, x[k])

        return predictions

        # raise NotImplementedError
