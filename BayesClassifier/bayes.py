import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

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
        # Obliczamy prawdopodobieństwa dla każdej klasy i próbki
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
            # KDE Gaussa dla klasy c (KernelDensity przyjmuje dane jako (n_samples, n_features))
            self.kdes_[c]       = KernelDensity(kernel="gaussian", bandwidth=self.h).fit(Xc)
        return self

    def predict_proba(self, X):
        # Obliczamy log-prawdopodobieństwa dla każdej klasy i próbki
        log_scores = np.zeros((len(X), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            prior = self.priors_[c]
            log_prior = np.log(prior + 1e-300)
            log_density = self.kdes_[c].score_samples(X)
            log_scores[:, i] = log_prior + log_density
            
        max_log = np.max(log_scores, axis=1, keepdims=True)
        scores = np.exp(log_scores - max_log)
        probs = scores / scores.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]