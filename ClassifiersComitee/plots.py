import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

os.makedirs("plots", exist_ok=True)

# 1 - Wykres zbiorow
def wykres_danych(X, y, nazwa):
    """Zadanie 1 – scatter danych. Dla >2 cech stosuje rzut PCA."""
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        X2 = PCA(n_components=2).fit_transform(X)
        podtytul = "(rzut PCA – pierwsze 2 składowe)"
    else:
        X2 = X
        podtytul = ""

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm", s=18, edgecolor="k", lw=0.3)
    ax.set_title(f"Dane: {nazwa}\n{podtytul}" if podtytul else f"Dane: {nazwa}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.tight_layout()
    _zapisz(fig, f"{nazwa}_1_dane.png")

# 2 - wykres doboru parametrow
def wykres_kandydatow(posortowani, Xtest, ytest, nazwa):

    # stała kolejność – zawsze ta sama na obu wykresach
    stala_kolejnosc = ["KNN-3", "KNN-5", "KNN-7", "Tree-3", "Tree-5", "SVM-0.5", "SVM-1", "Logistic"]
    slownik = {nk: (m, acc) for nk, m, acc in posortowani}

    etykiety, acc_dobor, acc_test = [], [], []
    for nk in stala_kolejnosc:
        if nk not in slownik:
            continue
        m, acc = slownik[nk]
        etykiety.append(nk)
        acc_dobor.append(acc)
        acc_test.append(accuracy_score(ytest, m.predict(Xtest)))

    x = np.arange(len(etykiety))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, acc_dobor, 0.35, label="zbiór doboru parametrów", color="steelblue")
    ax.bar(x + 0.2, acc_test,  0.35, label="zbiór testowy",           color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(etykiety, rotation=30, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Dobór parametrów klasyfikatora – {nazwa}")
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    _zapisz(fig, f"{nazwa}_2_pojedyncze.png")

# 3 - Wykres glosowan komitetow
def wykres_komitetu(wyniki_komitetow, typ, tytul, nazwa, numer):

    wiersze = sorted([(n, maj, waz, ada) for t, n, maj, waz, ada in wyniki_komitetow if t == typ])
    ns   = [w[0] for w in wiersze]
    majs = [w[1] for w in wiersze]
    wazs = [w[2] for w in wiersze]
    adas = [w[3] for w in wiersze]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ns, majs, "o-",  label="a) większość",    color="steelblue", linewidth=2)
    ax.plot(ns, wazs, "s--", label="b) ważone",       color="tomato",    linewidth=2)
    ax.plot(ns, adas, "^:",  label="c) adaptacyjne",  color="seagreen",  linewidth=2)
    ax.set_title(f"{tytul}\n({nazwa})")
    ax.set_xlabel("N – liczba klasyfikatorów w komitecie")
    ax.set_ylabel("Accuracy")
    ax.set_xticks([3, 5, 7])
    ax.set_ylim(0.5, 1.0)
    ax.legend(); ax.grid(alpha=0.4)
    fig.tight_layout()
    _zapisz(fig, f"{nazwa}_{numer}_{typ}.png")

# Zapis plikow
def _zapisz(fig, nazwa_pliku):
    sciezka = os.path.join("plots", nazwa_pliku)
    fig.savefig(sciezka)
    plt.close(fig)
    print(f"  Zapisano: {sciezka}")