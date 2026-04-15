import numpy as np
from scipy.stats import norm, gaussian_kde

class BayesParametric:
    """
    Klasyfikator Bayesa z estymatorem parametrycznym (rozkład normalny).
    """

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_  = {}
        self.means_   = {}
        self.covs_    = {}
        self.stds_    = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = len(Xc) / len(X)
            self.means_[c]  = Xc.mean(axis=0)
            self.covs_[c]   = np.cov(Xc.T, bias=False)
            # Odchylenie standardowe per-feature (ddof=1 — estymator nieobciążony)
            self.stds_[c]   = Xc.std(axis=0, ddof=1)
        return self

    def predict_proba(self, X):
        scores = np.zeros((len(X), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            prior = self.priors_[c]
            pdf_vals = norm.pdf(X, loc=self.means_[c], scale=self.stds_[c])
            class_likelihood = pdf_vals.prod(axis=1)
            scores[:, i] = prior * class_likelihood
        probs = scores / scores.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class BayesParzen:
    """
    Klasyfikator Bayesa z estymatorem nieparametrycznym - okna Parzena.
    Parametr h (bandwidth) steruje wygładzeniem estymatora.
    """

    def __init__(self, bandwidth=1.0):
        self.h = bandwidth

    def fit(self, X, y):
        self.classes_    = np.unique(y)
        self.priors_     = {}
        self.train_data_ = {}
        self.kdes_       = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c]     = len(Xc) / len(X)
            self.train_data_[c] = Xc
            # KDE Gaussa dla klasy c (dane w formacie (d, n_samples))
            self.kdes_[c]       = gaussian_kde(Xc.T, bw_method=self.h)
        return self

    def predict_proba(self, X):
        scores = np.zeros((len(X), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            prior = self.priors_[c]
            densities = self.kdes_[c](X.T)
            scores[:, i] = prior * densities
        probs = scores / scores.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]