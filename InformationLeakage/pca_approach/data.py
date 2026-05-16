import numpy as np
import scipy.io as sio
from sklearn.datasets import make_moons

RNG = 42

# 1 - Dane syntetyczne
def generuj_dane():
    X, y = make_moons(n_samples=1000, noise=0.30, random_state=RNG)
    return X, y

# 2 - Dane z lab1 dane7.mat
def wczytaj_mat(sciezki=("data7.mat", "dane7.mat")):
    for p in sciezki:
        if not __import__("os").path.exists(p):
            continue
        m = sio.loadmat(p)
        if "uczacy" not in m or "testowy" not in m:
            continue
        uczacy  = m["uczacy"][0, 0]
        testowy = m["testowy"][0, 0]
        # X jest (cechy × próbki) → transponujemy na (próbki × cechy)
        Xtr = uczacy["X"].T.astype(float)
        Xte = testowy["X"].T.astype(float)
        ytr = uczacy["D"].ravel().astype(int)
        yte = testowy["D"].ravel().astype(int)
        print(f"  Wczytano: {p}")
        print(f"  Uczący : {Xtr.shape[0]} próbek, {Xtr.shape[1]} cech")
        print(f"  Testowy: {Xte.shape[0]} próbek, {Xte.shape[1]} cech")
        return Xtr, ytr, Xte, yte
    return None, None, None, None


def wczytaj_csv(path, delimiter=',', label_index=-1, header=False):
    """
    Wczytuje dane z pliku CSV do macierzy cech X i wektora etykiet y.

    - `path`: ścieżka do pliku CSV
    - `delimiter`: separator (domyślnie ',')
    - `label_index`: indeks kolumny z etykietami; domyślnie -1 (ostatnia kolumna)
    - `header`: czy plik ma nagłówek (True/False)

    Zwraca (X, y) jako `(np.ndarray, np.ndarray)` lub `(None, None)` jeśli błąd.
    """
    try:
        skip = 1 if header else 0
        data = np.genfromtxt(path, delimiter=delimiter, skip_header=skip)
        if data is None or data.size == 0:
            return None, None
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if label_index == -1:
            X = data[:, :-1]
            y = data[:, -1]
        else:
            X = np.delete(data, label_index, axis=1)
            y = data[:, label_index]
        return X.astype(float), y
    except Exception:
        return None, None