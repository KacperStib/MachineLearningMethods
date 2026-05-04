import select
from statistics import variance
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from data import generuj_dane, wczytaj_mat, wczytaj_csv
import glob
import os


def Metoda1(X, y, n_components=2):
    """
    Selekcja cech na pełnym zbiorze (fit i transform na całych danych),
    następnie trenowanie i testowanie na tych samych danych.
    Zwraca dokładność na tym samym zbiorze.
    """
    selector = PCA(n_components=n_components).fit(X)
    eig = plot_pca_eigenvalues(selector, show=True)
    X_sel = selector.transform(X)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_sel, y)
    acc = clf.score(X_sel, y)
    return acc


def Metoda2(X, y, n_components=2, test_size=0.3, random_state=42):
    """
    Najpierw selekcja cech dopasowana do całego zbioru (fit na X),
    potem podział na zbiór treningowy i testowy (transformujemy X przed podziałem).
    Zwraca dokładność na zbiorze testowym.
    """
    selector = PCA(n_components=n_components).fit(X)
    X_sel = selector.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X_sel, y, test_size=test_size, random_state=random_state)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    return acc


def Metoda3(X, y, n_components=2, test_size=0.3, random_state=42):
    """
    Najpierw podział na trening/test, następnie selekcja cech dopasowana
    tylko na zbiorze treningowym. Transformacja zbioru testowego odbywa się
    za pomocą selektora dopasowanego do danych treningowych.
    Zwraca dokładność na zbiorze testowym.
    """
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
    selector = PCA(n_components=n_components).fit(Xtr)
    Xtr_sel = selector.transform(Xtr)
    Xte_sel = selector.transform(Xte)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(Xtr_sel, ytr)
    acc = clf.score(Xte_sel, yte)
    return acc


def plot_pca_eigenvalues(pca, show=True, save_path=None):

    eigvals = pca.explained_variance_
    indices = np.arange(1, len(eigvals) + 1)
    plt.figure(figsize=(8, 4))
    plt.bar(indices, eigvals, color="C0")
    plt.plot(indices, eigvals, marker="o", color="C1")
    plt.xlabel("Index (principal component)")
    plt.ylabel("Eigenvalue (explained variance)")
    plt.title(f"PCA eigenvalues — first {len(eigvals)} components")
    plt.xticks(indices)
    for i, v in enumerate(eigvals):
        plt.text(indices[i], v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    return eigvals

if __name__ == "__main__":
    # Spróbuj wczytać pierwszy plik CSV z katalogu `dane/`, jeśli istnieje.
    csv_files = glob.glob(os.path.join("dane", "*.csv"))
    if csv_files:
        x, y = wczytaj_csv(csv_files[0], delimiter=',', header=True)
        if x is None:
            print(f"Nie udało się wczytać {csv_files[0]}, używam danych syntetycznych.")
            x, y = generuj_dane()
        else:
            print(f"Wczytano dane z {csv_files[0]} -> X:{x.shape}, y:{y.shape}")
    else:
        x, y = generuj_dane()
    n_components = 8
    a1 = Metoda1(x, y, n_components)
    a2 = Metoda2(x, y, n_components, test_size=0.2, random_state=42)
    a3 = Metoda3(x, y, n_components, test_size=0.2, random_state=42)

    print(f"Metoda1 (selekcja -> tren/test na tych samych danych): acc={a1:.4f}")
    print(f"Metoda2 (selekcja na całym zbiorze -> podział): acc={a2:.4f}")
    print(f"Metoda3 (podział -> selekcja na treningu -> test): acc={a3:.4f}")