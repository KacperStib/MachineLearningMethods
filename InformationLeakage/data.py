"""
data.py – wczytywanie danych, helpery selekcji cech i klasyfikatora, wykres
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier

# ===========================================================================
# PARAMETRY
# ===========================================================================
SET1_FILE        = "dane/S6_set1.csv"
SET2_FILE        = "dane/S6_set2.csv"
LABELS_FILE      = "dane/labels.csv"
EXT_SET_FILE     = "dane/sonardata.csv"  # dodatkowy zbiór z 60 cechami i etykietami 'R'/'M'
K_FEATURES       = 5
K_NEIGHBORS      = 3
TEST_SIZE        = 0.5
RANDOM_SEED      = 42
N_MCCV           = 100
STABILITY_THRESH = 0.5

# ===========================================================================
# WCZYTANIE DANYCH
# ===========================================================================
def load_dataset(data_file: str, labels_file: str = LABELS_FILE):
    X = pd.read_csv(data_file, header=None).values
    y = pd.read_csv(labels_file, header=None).values.ravel()
    print(f"Załadowano: {data_file}")
    print(f"  Próbki : {X.shape[0]}  (klas 0: {(y==0).sum()}, klas 1: {(y==1).sum()})")
    print(f"  Cechy  : {X.shape[1]}")
    return X, y

def load_dataset2(data_file: str):
    """
    Wczytuje dane z pliku sonardata.csv (60 cech + etykieta tekstowa 'R'/'M').
    """
    df = pd.read_csv(data_file, header=None)
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values
    # Konwersja etykiet tekstowych na numeryczne (np. R=0, M=1)
    y = np.where(y_raw == 'R', 0, 1)
    
    print(f"Załadowano: {data_file}")
    print(f"  Próbki : {X.shape[0]}  (klas 0 (R): {(y==0).sum()}, klas 1 (M): {(y==1).sum()})")
    print(f"  Cechy  : {X.shape[1]}")
    return X, y

# ===========================================================================
# HELPERS – selekcja cech
# ===========================================================================
def get_clf():
    return KNeighborsClassifier(n_neighbors=K_NEIGHBORS)

def select_on_all(X, y, k=K_FEATURES):
    """Selekcja na CAŁYM zbiorze – powoduje wyciek (Scen. 1 i 2)."""
    sel = SelectKBest(f_classif, k=k)
    X_sel = sel.fit_transform(X, y)
    return X_sel, sel.get_support(indices=True)

def select_on_train(X_train, y_train, X_test, k=K_FEATURES):
    """Selekcja TYLKO na train – brak wycieku (Scen. 3)."""
    sel = SelectKBest(f_classif, k=k)
    X_train_sel = sel.fit_transform(X_train, y_train)
    X_test_sel  = sel.transform(X_test)
    return X_train_sel, X_test_sel, sel.get_support(indices=True)

# ===========================================================================
# DEMONSTRACJA – punkt a)
# ===========================================================================
def demo_feature_selection(X, y, k=K_FEATURES):
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    scores   = selector.scores_
    pvalues  = selector.pvalues_
    selected = selector.get_support(indices=True)
    print(f"\nANOVA F-score – top {k} cech:")
    print(f"  Indeksy wybranych cech : {selected.tolist()}")
    print(f"  F-score (wybrane)      : {scores[selected].round(2).tolist()}")
    print(f"  p-value (wybrane)      : {pvalues[selected].round(4).tolist()}")
    return selector, selected

def demo_classifier(X_sel, y):
    clf = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    clf.fit(X_sel, y)
    print(f"\nKNN (k={K_NEIGHBORS}) – wytrenowany na {X_sel.shape[0]} próbkach, "
          f"{X_sel.shape[1]} cechach")
    print(f"  Klasy: {clf.classes_}")
    return clf

# ===========================================================================
# WYKRES – istotność cech
# ===========================================================================
def plot_feature_importance(X, y, title):
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    scores  = selector.scores_
    indices = np.argsort(scores)[::-1]
    n_plot  = min(20, X.shape[1])

    plt.figure(figsize=(12, 5))
    plt.bar(range(n_plot), scores[indices[:n_plot]], color='crimson', alpha=0.7)
    plt.xticks(range(n_plot), [f"C{i}" for i in indices[:n_plot]], rotation=45)
    plt.title(f"Istotność cech (Top 20) – {title}")
    plt.ylabel("F-score")
    plt.xlabel("Indeks cechy")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()