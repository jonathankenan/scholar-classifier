import numpy as np
import pandas as pd
from core.base_model import BaseClassifier

class SoftmaxRegression(BaseClassifier):
    """
    Multinomial logistic regression (softmax regression) classifier.
    
    This model:
    - Learns a separate weight vector for each output class.
    - Uses the softmax function to produce a probability distribution
      over all classes for each input sample.
    - Is trained using gradient-based optimization on the categorical
      cross-entropy loss.
    """
    
    def __init__(self, lr=0.1, max_iter=100, n_classes=None, verbose=False):
        # Hyperparameters
        self.lr = lr
        self.max_iter = max_iter
        self.n_classes = n_classes
        self.verbose = verbose

        # Learned during training
        self.weights = None  # shape (n_features, n_classes)
        self.mean = None
        self.std = None
        self.loss_history = []
        self.classes_ = None

    def _add_bias(self, X: np) -> np.ndarray:
        """
        Add a bias feature (column of ones) to the input data X.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix without bias.

        Returns
        -------
        X_bias : np.ndarray, shape (n_samples, n_features + 1)
            Input matrix with an additional bias column at the front.
        """
        # TODO: implement bias term concatenation
        raise NotImplementedError("Method _add_bias not implemented yet")
    
    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute the softmax probabilities for each class.
        
        Parameters
        ----------
        Z : np.ndarray, shape (n_samples, n_classes)
            The linear scores for each class.

        Returns
        -------
        probs : np.ndarray, shape (n_samples, n_classes)
            The softmax probabilities for each class.
        
        Notes
        -----
        The softmax function is defined as:
            softmax(z_i) = exp(z_i) / sum_j exp(z_j)
        """
        # TODO: implement softmax computation
        raise NotImplementedError("Method _softmax not implemented yet")
    
    def _one_hot(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        """
        One-hot encode the class labels.
        
        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            Class labels as integers.
        n_classes : int
            Total number of classes.

        Returns
        -------
        y_one_hot : np.ndarray, shape (n_samples, n_classes)
            One-hot encoded class labels.
        """
        # TODO: implement one-hot encoding
        raise NotImplementedError("Method _one_hot not implemented yet")
    
    def _compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy loss.
        
        Parameters
        ----------
        Y_true : np.ndarray, shape (n_samples, n_classes)
            One-hot encoded true class labels.
        Y_pred : np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities.

        Returns
        -------
        loss : float
            The average cross-entropy loss over all samples.
        """
        # TODO: implement loss computation
        raise NotImplementedError("Method _compute_loss not implemented yet")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the softmax regression model using gradient descent.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.
        y : np.ndarray, shape (n_samples,)
            Class labels as integers.

        Steps
        -----
        1. Optionally standardize features and store mean and std.
        2. Add bias term to X.
        3. Initialize weight matrix W.
        4. Convert y to one-hot representation.
        5. For each epoch:
            a. Compute linear scores Z = X @ W.T
            b. Compute probabilities P = softmax(Z)
            c. Compute error (Y_true - P)
            d. Update W using gradient ascent with learning rate.
            e. Optionally compute and store loss for monitoring.
        """
        # TODO: implement training procedure
        raise NotImplementedError("Method fit not implemented yet")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        probs : np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities for each sample.
        """
        # TODO: implement probability prediction
        raise NotImplementedError("Method predict_proba not implemented yet")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels as integers.
        """
        # TODO: implement class label prediction
        raise NotImplementedError("Method predict not implemented yet")