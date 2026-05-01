import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RNG = 42

# =============================================================
# Zadanie 2 – kandydaci do doboru parametrów
# =============================================================

def kandydaci():
    """Słownik modeli do porównania na zbiorze doboru parametrów."""
    def knn(k):
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))

    def svm(C):
        return make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf", probability=True))

    def logreg():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    return {
        "KNN-3":    knn(3),
        "KNN-5":    knn(5),
        "KNN-7":    knn(7),
        "Tree-3":   DecisionTreeClassifier(max_depth=3, random_state=RNG),
        "Tree-5":   DecisionTreeClassifier(max_depth=5, random_state=RNG),
        "SVM-0.5":  svm(0.5),
        "SVM-1":    svm(1.0),
        "Logistic": logreg(),
    }


def ocen_kandydatow(Xtr, ytr, Xval, yval):
    wyniki = []
    for nazwa, model in kandydaci().items():
        m = clone(model).fit(Xtr, ytr)
        y_pred = m.predict(Xval)
        acc = accuracy_score(yval, y_pred)
        wyniki.append((nazwa, m, acc))
    return sorted(wyniki, key=lambda t: t[2], reverse=True)


# =============================================================
# Zadanie 3 – budowanie komitetów
# =============================================================

def komitet_bagging(N, Xtr, ytr, Xval, yval):
    model_bazowy = DecisionTreeClassifier(max_depth=5, random_state=RNG)
    rng = np.random.default_rng(RNG)
    czlonkowie, wagi = [], []
    for _ in range(N):
        idx = rng.integers(0, len(Xtr), len(Xtr))      # losowanie ze zwracaniem
        m = clone(model_bazowy).fit(Xtr[idx], ytr[idx])
        czlonkowie.append(m)
        wagi.append(accuracy_score(yval, m.predict(Xval)))
    return czlonkowie, np.array(wagi)


def komitet_roznorodny(posortowani_kandydaci, N, Xtr, ytr, Xval, yval):
    """
    3b – Różne klasyfikatory, te same dane.
    Bierzemy N najlepszych kandydatów i douczamy na pełnym Xtr.
    """
    czlonkowie, wagi = [], []
    for _, model, _ in posortowani_kandydaci[:N]:
        m = clone(model).fit(Xtr, ytr)
        czlonkowie.append(m)
        wagi.append(accuracy_score(yval, m.predict(Xval)))
    return czlonkowie, np.array(wagi)


def komitet_knn_param(N, Xtr, ytr, Xval, yval):
    """
    3c – Różne parametry (k) dla KNN, te same dane.
    """
    ks = [1, 3, 5, 7, 9, 11, 15][:N]
    czlonkowie, wagi = [], []
    for k in ks:
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        m.fit(Xtr, ytr)
        czlonkowie.append(m)
        wagi.append(accuracy_score(yval, m.predict(Xval)))
    return czlonkowie, np.array(wagi)


# =============================================================
# Głosowanie
# =============================================================
# 3a
def glosowanie_wiekszosci(czlonkowie, X):
    """Każdy klasyfikator oddaje głos - wygrywa klasa z największą liczbą głosów."""
    predykcje = np.vstack([m.predict(X) for m in czlonkowie])
    return np.apply_along_axis(
        lambda kol: np.bincount(kol.astype(int)).argmax(), axis=0, arr=predykcje
    )

# 3b
def glosowanie_wazone(czlonkowie, wagi, X):
    """Głosy ważone accuracy członka na zbiorze doboru parametrów."""
    w = wagi / wagi.sum() if wagi.sum() > 0 else np.ones(len(wagi)) / len(wagi)

    # Preferujemy prawdopodobieństwa (soft voting)
    proba_list = [m.predict_proba(X)[:, 1] for m in czlonkowie if hasattr(m, "predict_proba")]
    if proba_list:
        agregat = np.average(np.vstack(proba_list), axis=0, weights=w[:len(proba_list)])
        return (agregat >= 0.5).astype(int)

    # Fallback – dyskretne głosowanie ważone
    predykcje = np.vstack([m.predict(X) for m in czlonkowie])
    n_klas = int(predykcje.max()) + 1
    skumulowane = np.zeros((X.shape[0], n_klas))
    for wi, p in zip(w, predykcje):
        for i, cls in enumerate(p):
            skumulowane[i, int(cls)] += wi
    return np.argmax(skumulowane, axis=1)

# 3c
def glosowanie_adaptacyjne(czlonkowie, wagi_per_klasa, X):
    predykcje = np.vstack([m.predict(X) for m in czlonkowie])  # (N, n_probek)
    n_klas = wagi_per_klasa[0].shape[0]
    skumulowane = np.zeros((X.shape[0], n_klas))

    for i, (p, w_klas) in enumerate(zip(predykcje, wagi_per_klasa)):
        for j, cls in enumerate(p):
            skumulowane[j, int(cls)] += w_klas[int(cls)]

    return np.argmax(skumulowane, axis=1)

# dobierz wage dla klasy z recall 
# klasyfikator ma wieksza wage dla klasy ktora lepiej rozpoznaje
def wagi_per_klasa_z_doboru(czlonkowie, Xval, yval):
    klasy = np.unique(yval)
    eps = 1e-6
    wynik = []
    for m in czlonkowie:
        pred = m.predict(Xval)
        recall = np.array([
            np.mean(pred[yval == k] == k) if (yval == k).any() else 0.0
            for k in klasy
        ])
        blad = 1.0 - recall
        wagi_klas = np.array([
            recall[i] / (blad.sum() - blad[i] + eps)
            for i in range(len(klasy))
        ])
        wynik.append(wagi_klas)
    return wynik

# Dokladnosc
def ocen_komitet(czlonkowie, wagi, Xval, yval, Xtest, ytest):
    """Zwraca (acc_większość, acc_ważone, acc_adaptacyjne) na zbiorze testowym."""
    pred_maj = glosowanie_wiekszosci(czlonkowie, Xtest)
    pred_waz = glosowanie_wazone(czlonkowie, wagi, Xtest)

    wagi_klas = wagi_per_klasa_z_doboru(czlonkowie, Xval, yval)
    pred_ada  = glosowanie_adaptacyjne(czlonkowie, wagi_klas, Xtest)

    return (
        accuracy_score(ytest, pred_maj),
        accuracy_score(ytest, pred_waz),
        accuracy_score(ytest, pred_ada),
        (ytest, pred_maj),
        (ytest, pred_waz),
        (ytest, pred_ada)
    )