import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

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

    etykiety, acc_dobor, acc_test, y_pred = [], [], [], []
    for nk in stala_kolejnosc:
        if nk not in slownik:
            continue
        m, acc = slownik[nk]
        etykiety.append(nk)
        acc_dobor.append(acc)
        y_pred.append(m.predict(Xtest))
        acc_test.append(accuracy_score(ytest, y_pred[-1]))

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

    fig2, ax2 = plt.subplots(3, 3, figsize=(8, 6))
    ax2 = ax2.flatten()
    for i, (label, ypred) in enumerate(zip(etykiety, y_pred)):
        ConfusionMatrixDisplay.from_predictions(ytest, ypred, ax=ax2[i], cmap="Blues")
        ax2[i].set_title(label)
    for j in range(len(etykiety), len(ax2)):
        ax2[j].axis("off")
    fig2.tight_layout()
    _zapisz(fig2, f"{nazwa}_2_cm.png")

# 3 - Wykres glosowan komitetow
def wykres_komitetu(wyniki_komitetow, typ, tytul, nazwa, numer):

    wiersze = sorted([(n, maj, waz, ada) for t, n, maj, waz, ada, _, _, _ in wyniki_komitetow if t == typ])
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

    wiersze_full = sorted([
        (n, maj, waz, ada, y_maj, y_waz, y_ada)
        for t, n, maj, waz, ada, y_maj, y_waz, y_ada in wyniki_komitetow
        if t == typ
    ])
    if not wiersze_full:
        return

    rows = len(wiersze_full)
    fig2, ax2 = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    ax2 = np.atleast_2d(ax2)

    for i, (n, maj, waz, ada, y_maj, y_waz, y_ada) in enumerate(wiersze_full):
        y_true_maj, y_pred_maj = y_maj
        y_true_waz, y_pred_waz = y_waz
        y_true_ada, y_pred_ada = y_ada

        ConfusionMatrixDisplay.from_predictions(y_true_maj, y_pred_maj, ax=ax2[i, 0], cmap="Blues")
        ax2[i, 0].set_title(f"N={n} majority")

        ConfusionMatrixDisplay.from_predictions(y_true_waz, y_pred_waz, ax=ax2[i, 1], cmap="Blues")
        ax2[i, 1].set_title(f"N={n} weighted")

        ConfusionMatrixDisplay.from_predictions(y_true_ada, y_pred_ada, ax=ax2[i, 2], cmap="Blues")
        ax2[i, 2].set_title(f"N={n} adaptive")

    fig2.tight_layout()
    _zapisz(fig2, f"{nazwa}_{numer}_{typ}_cm.png")

# Zapis plikow
def _zapisz(fig, nazwa_pliku):
    sciezka = os.path.join("plots", nazwa_pliku)
    fig.savefig(sciezka)
    plt.close(fig)
    print(f"  Zapisano: {sciezka}")