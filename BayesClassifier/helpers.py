import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap

# Metryki
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix_2x2(y_true, y_pred, classes=(0, 1)):
    cm = np.zeros((2, 2), dtype=int)
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            cm[i, j] = np.sum((y_true == ci) & (y_pred == cj))
    return cm

# Jakosc klasyfikatora
def classification_metrics(y_true, y_pred):
    """
    Zwraca slownik metryk na podstawie macierzy pomylek.
    Klasa pozytywna = 1 (A2), klasa negatywna = 0 (A1).
 
        Pred A1(0)  Pred A2(1)
    True A1(0)   TN          FP
    True A2(1)   FN          TP
    """
    cm = confusion_matrix_2x2(y_true, y_pred)
    TN, FP = int(cm[0, 0]), int(cm[0, 1])
    FN, TP = int(cm[1, 0]), int(cm[1, 1])
 
    acc         = (TP + TN) / (TP + FP + TN + FN)
    precision   = TP / (TP + FP)  if (TP + FP) > 0 else 0.0
    recall      = TP / (TP + FN)  if (TP + FN) > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
    specificity = TN / (TN + FP)  if (TN + FP) > 0 else 0.0
 
    return dict(TP=TP, FP=FP, TN=TN, FN=FN,
                ACC=acc, precision=precision,
                recall=recall, specificity=specificity, F1=f1, cm=cm)
 
 # Raport po uczeniu
def print_report(name, y_true, y_pred):
    m = classification_metrics(y_true, y_pred)
    print(f"\n{'─'*52}")
    print(f" {name}")
    print(f"{'─'*52}")
    print(f"  Macierz pomylek (klasa pozytywna = A2):")
    print(f"               Pred A1   Pred A2")
    print(f"  True A1  TN= {m['TN']:5d}   FP= {m['FP']:5d}")
    print(f"  True A2  FN= {m['FN']:5d}   TP= {m['TP']:5d}")
    print(f"\n  ACC       = (TP+TN)/(TP+FP+TN+FN) = {m['ACC']*100:.2f}%")
    print(f"  Precyzja  = TP/(TP+FP)             = {m['precision']*100:.2f}%")
    print(f"  Czulosc   = TP/(TP+FN)             = {m['recall']*100:.2f}%")
    print(f"  Swoistosc = TN/(TN+FP)             = {m['specificity']*100:.2f}%")
    print(f"  F1        = 2*P*R/(P+R)            = {m['F1']*100:.2f}%")
    return m

# Wykresy
COLORS = {0: "#E63946", 1: "#1D6A96"}

# Elipsa odchylenia standarowego zbioru
def plot_cov_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """Rysuje elipsę odpowiadającą n_std odchyleń standardowych."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=w, height=h, angle=angle, **kwargs)
    ax.add_patch(ellipse)

# Wykres zbioru uczacego i testowego (przed Bayesem)
def plot_dataset(axes, X_train, y_train, X_test, y_test,
                 mu1, Sigma1, mu2, Sigma2):
    """Rysuje zbiory uczący i testowy z elipsami kowariancji."""
    labels = {0: "A1", 1: "A2"}
    for ax, (X_sub, y_sub, title) in zip(
            axes,
            [(X_train, y_train, "Zbiór uczący (100 pkt)"),
             (X_test,  y_test,  "Zbiór testowy (100 pkt)")]):

        for cls in [0, 1]:
            mask = y_sub == cls
            ax.scatter(X_sub[mask, 0], X_sub[mask, 1],
                       c=COLORS[cls], label=f"Klasa {labels[cls]}",
                       alpha=0.75, edgecolors="white", linewidths=0.4,
                       s=55, zorder=3)

        for cls, mu, Sigma in [(0, mu1, Sigma1), (1, mu2, Sigma2)]:
            for n_std, alpha in [(1, 0.25), (2, 0.12)]:
                plot_cov_ellipse(ax, mu, Sigma, n_std=n_std,
                                 facecolor=COLORS[cls], alpha=alpha,
                                 edgecolor=COLORS[cls], linewidth=1.5)
            ax.plot(*mu, marker="x", color=COLORS[cls],
                    markersize=10, markeredgewidth=2, zorder=4)

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("Cecha 1", fontsize=10)
        ax.set_ylabel("Cecha 2", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(-5, 7); ax.set_ylim(-5, 7)
        ax.set_aspect("equal")

# Wykres granicy decyzyjnej na wykresie danych
def plot_decision_boundary(ax, clf, X, y, title, h=0.05):
    """Rysuje granicę decyzyjną i punkty danych."""
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    cmap_bg = ListedColormap(["#FFCDD2", "#BBDEFB"])
    ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.55)
    ax.contour (xx, yy, Z, colors="black", linewidths=0.8, linestyles="--")

    for c, lbl in zip([0, 1], ["A1", "A2"]):
        mask = y == c
        ax.scatter(X[mask,0], X[mask,1], c=COLORS[c], label=f"Klasa {lbl}",
                   edgecolors="white", linewidths=0.4, s=45, zorder=3)

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("Cecha 1"); ax.set_ylabel("Cecha 2")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.3)

# Wykres przy porownywaniu szerokosci okna - h
def plot_accuracy_vs_bandwidth(parzen_results, param_train_acc, param_test_acc):
    """Wykres dokładności klasyfikatora Parzena w funkcji szerokości okna h."""
    hs      = list(parzen_results.keys())
    accs_tr = [parzen_results[h][0]*100 for h in hs]
    accs_te = [parzen_results[h][1]*100 for h in hs]

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(hs, accs_tr, "o-",  color="#E63946", linewidth=2,
            markersize=7, label="Parzen – uczący")
    ax.plot(hs, accs_te, "s--", color="#1D6A96", linewidth=2,
            markersize=7, label="Parzen – testowy")
    ax.axhline(param_train_acc*100, color="#E63946", linestyle=":", linewidth=1.5,
               label=f"Param. – uczący  ({param_train_acc*100:.1f}%)")
    ax.axhline(param_test_acc*100,  color="#1D6A96", linestyle=":", linewidth=1.5,
               label=f"Param. – testowy ({param_test_acc*100:.1f}%)")
    ax.set_xlabel("Szerokość okna  h", fontsize=11)
    ax.set_ylabel("Dokładność [%]",    fontsize=11)
    ax.set_title("Wpływ szerokości okna Parzena na dokładność klasyfikatora",
                 fontsize=11)
    ax.set_xscale("log")
    ax.set_ylim(60, 102)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, framealpha=0.9)
    return fig

# Tabela zbiorcza
def compare_results_table(results):
    """
    Drukuje zbiorczą tabelę porównawczą wyników.
    results: lista slownikow {'name': str, 'metrics': dict_z_classification_metrics}
    """
    print(f"\n{'═'*78}")
    print(f" TABELA POROWNAWCZA – zbior testowy")
    print(f"{'═'*78}")
    header = f"{'Klasyfikator':<35} {'ACC':>7} {'Precyzja':>9} {'Czulosc':>8} {'F1':>7}"
    print(header)
    print("─"*78)
    for r in results:
        m = r['metrics']
        print(f"  {r['name']:<33} {m['ACC']*100:>6.2f}%  "
              f"{m['precision']*100:>8.2f}%  "
              f"{m['recall']*100:>7.2f}%  "
              f"{m['F1']*100:>6.2f}%")
    print("═"*78)

# 5 cech
def plot_features_pairwise(X, y, n_show=5, title=""):
    """Scatter-plot pierwszych n_show cech parami (macierz wykresow)."""
    d    = min(n_show, X.shape[1])
    fig, axes = plt.subplots(d, d, figsize=(2.5*d, 2.5*d), constrained_layout=True)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                for c in [0, 1]:
                    ax.hist(X[y==c, i], bins=20, color=COLORS[c],
                            alpha=0.55, density=True)
                ax.set_ylabel(f"C{i+1}", fontsize=7)
            else:
                for c in [0, 1]:
                    mask = y == c
                    ax.scatter(X[mask,j], X[mask,i], c=COLORS[c],
                               s=8, alpha=0.5, linewidths=0)
            ax.tick_params(labelsize=6)
            if i == d-1: ax.set_xlabel(f"C{j+1}", fontsize=7)
    return fig    