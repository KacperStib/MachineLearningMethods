import numpy as np
import scipy.io

# Generacja danych
def generate_data(seed=42):
    np.random.seed(seed)

    # Dane z instrukcji
    mu1    = np.array([0, 0])
    Sigma1 = np.array([[2, -1], [-1, 2]])

    mu2    = np.array([2, 2])
    Sigma2 = np.array([[1,  0], [ 0, 1]])

    # Ilosc probek
    N = 100

    # syntetyczny zbior danych
    X_A1 = np.random.multivariate_normal(mu1, Sigma1, N)
    X_A2 = np.random.multivariate_normal(mu2, Sigma2, N)

    X = np.vstack([X_A1, X_A2])
    y = np.array([0]*N + [1]*N)

    return X, y, mu1, Sigma1, mu2, Sigma2

# Podzial danych na zbior do uczenia i zbior testowy
def train_test_split(X, y, train_size=100):
    idx       = np.random.permutation(len(X))
    train_idx = idx[:train_size]
    test_idx  = idx[train_size:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

## Zadanie 3 - zbiory po 5 cech
def generate_data_gaussian(n_features=5, N=100, sep=2.0, seed=42):
    """
    Zbior wielocechowy - obie klasy z rozkladu normalnego.
    Klasa A1: mu=0, Sigma ze skorelowanymi cechami (rho=0.6).
    Klasa A2: mu=sep, Sigma = I  (cechy niezalezne).
    Dobry przypadek dla klasyfikatora PARAMETRYCZNEGO.
    """
    np.random.seed(seed)
    d = n_features
 
    mu1    = np.zeros(d)
    rho    = 0.6
    Sigma1 = np.full((d, d), rho)
    np.fill_diagonal(Sigma1, 1.0)
 
    mu2    = np.full(d, sep)
    Sigma2 = np.eye(d)
 
    X_A1 = np.random.multivariate_normal(mu1, Sigma1, N)
    X_A2 = np.random.multivariate_normal(mu2, Sigma2, N)
 
    X = np.vstack([X_A1, X_A2])
    y = np.array([0]*N + [1]*N)
    return X, y
 
 
def generate_data_nongaussian(n_features=5, N=100, seed=42):
    """
    Zbior wielocechowy - cechy z rozkladu NIEnormalnego.
    Klasa A1: mieszanina 2 gaussianow (bimodalna) na kazdej cesze.
    Klasa A2: rozklad jednostajny na [0, 4].
    Dobry przypadek dla klasyfikatora PARZENA (nieparametrycznego).
    """
    np.random.seed(seed)
    d = n_features
 
    half   = N // 2
    X_A1_a = np.random.normal(-2.0, 0.5, (half,     d))
    X_A1_b = np.random.normal( 2.0, 0.5, (N - half, d))
    X_A1   = np.vstack([X_A1_a, X_A1_b])
 
    X_A2 = np.random.uniform(0.0, 4.0, (N, d))
 
    X = np.vstack([X_A1, X_A2])
    y = np.array([0]*N + [1]*N)
    return X, y

# Zadnie 4 - wczytanie klas nowotworow
"""
Struktura pliku:
  dane7.mat zawiera dwie struktury: 'uczacy' i 'testowy'
  Kazda ma pola:
    X : (300 x N)  – ekspresja genow  (wiersze=geny, kolumny=probki)
    D : (1  x N)   – etykiety klas    0=lagodny, 1=rak brodawkowaty
"""
 
def load_microarray(mat_path):
    """
    Wczytuje dane z jednego pliku .mat zawierajacego struktury 'uczacy' i 'testowy'.
    Zwraca:
        X_train (85  x 300) – wiersze=probki, kolumny=geny
        y_train (85,)
        X_test  (40  x 300)
        y_test  (40,)
    """
    mat = scipy.io.loadmat(mat_path)
 
    uczacy  = mat['uczacy'][0, 0]
    testowy = mat['testowy'][0, 0]
 
    # X jest (300 x N) → transponujemy na (N x 300)
    X_train = uczacy['X'].T.astype(float)     # (85,  300)
    y_train = uczacy['D'].ravel().astype(int)  # (85,)
 
    X_test  = testowy['X'].T.astype(float)    # (40,  300)
    y_test  = testowy['D'].ravel().astype(int) # (40,)
 
    print(f"Zbior uczacy : {X_train.shape[0]:3d} probek, {X_train.shape[1]} genow")
    print(f"  Klasa 0 (lagodny): {(y_train==0).sum()}")
    print(f"  Klasa 1 (rak):     {(y_train==1).sum()}")
    print(f"Zbior testowy: {X_test.shape[0]:3d} probek, {X_test.shape[1]} genow")
    print(f"  Klasa 0 (lagodny): {(y_test==0).sum()}")
    print(f"  Klasa 1 (rak):     {(y_test==1).sum()}")
 
    return X_train, y_train, X_test, y_test
 