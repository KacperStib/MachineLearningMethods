import numpy as np
from scipy.stats import norm

class BayesParametric:
    """
    Klasyfikator Bayesa z estymatorem parametrycznym (rozkład normalny).
    Estymuje μ i Σ (pełna macierz kowariancji) z danych uczących.
    Decyzja: argmax_k  log P(k) + log N(x | μ_k, Σ_k)
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
    Jądro: wielowymiarowy Gauss  K(u) = (2π)^{-d/2} exp(-||u||²/2)
    Gęstość: p(x) = (1/n) Σ_i (1/h^d) K((x - x_i)/h)
    Parametr h (bandwidth) steruje wygładzeniem estymatora.
    """

    def __init__(self, bandwidth=1.0):
        self.h = bandwidth

    def fit(self, X, y):
        self.classes_    = np.unique(y)
        self.priors_     = {}
        self.train_data_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c]     = len(Xc) / len(X)
            self.train_data_[c] = Xc
        return self

    def _parzen_density(self, x, X_train):
        """KDE z jądrem gaussowskim dla punktu x."""
        n, d = X_train.shape
        diff = (x - X_train) / self.h
        log_kernels = -0.5 * np.sum(diff**2, axis=1)
        lmax = log_kernels.max()
        log_density = (lmax
                       + np.log(np.exp(log_kernels - lmax).sum())
                       - np.log(n)
                       - d * np.log(self.h)
                       - 0.5 * d * np.log(2*np.pi))
        return log_density

    def predict_proba(self, X):
        log_posts = np.zeros((len(X), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            log_prior = np.log(self.priors_[c])
            log_posts[:, i] = np.array([
                log_prior + self._parzen_density(x, self.train_data_[c])
                for x in X
            ])
        log_posts -= log_posts.max(axis=1, keepdims=True)
        probs = np.exp(log_posts)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]