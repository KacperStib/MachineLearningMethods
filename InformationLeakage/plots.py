"""
plots.py – wykresy do sprawozdania
Pokrycie punktów: b, c, e, f, g
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from data import STABILITY_THRESH

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===========================================================================
# PUNKT a) – Istotność cech (F-score) – uzasadnienie wyboru k=5
# ===========================================================================
def plot_feature_importance(X, y, title, k_selected=5):
    """
    Wykres F-score top 20 cech.
    Zaznacza pionową linią wybrane k cech – uzasadnienie 'efektu łokcia'.
    Używany w punkcie a) i b).
    """
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    scores  = selector.scores_
    indices = np.argsort(scores)[::-1]
    n_plot  = min(20, X.shape[1])

    plt.figure(figsize=(12, 5))
    bars = plt.bar(range(n_plot), scores[indices[:n_plot]], color='crimson', alpha=0.7)

    # Zaznacz wybrane k cech (tylko te które mieszczą się na wykresie)
    n_highlight = min(k_selected, n_plot)
    for i in range(n_highlight):
        bars[i].set_color('#8B0000')
        bars[i].set_alpha(1.0)
    plt.axvline(x=k_selected - 0.5, color='navy', linestyle='--',
                linewidth=1.5, label=f'próg k={k_selected}')

    plt.xticks(range(n_plot), [f"C{indices[i]}" for i in range(n_plot)], rotation=45)
    plt.title(f"Istotność cech (Top 20) – {title}\n"
              f"Ciemne słupki = wybrane {k_selected} cech")
    plt.ylabel("F-score (ANOVA)")
    plt.xlabel("Indeks cechy")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"plot_feature_importance_{title}.png"), dpi=150)
    plt.show()

# ===========================================================================
# Boxplot błędów MCCV dla obu zbiorów
# Pokazuje różnicę między scenariuszami i między zbiorami
# ===========================================================================
def plot_mccv_boxplot(wyniki_mccv):
    """
    Boxplot rozkładu błędów MCCV dla wszystkich 3 scenariuszy i obu zbiorów.
    Kluczowy wykres do punktów c) i e) – porównanie wyników.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Rozkład błędów klasyfikacji – MCCV (100 iteracji)\n"
                 "Porównanie scenariuszy walidacji", fontsize=13, fontweight='bold')

    colors = ['#E24B4A', '#EF9F27', '#2196A6']
    labels = ['Scen. 1\nResubstytucja', 'Scen. 2\nLeaky holdout', 'Scen. 3\nPoprawny holdout']

    for ax, (nazwa, (e1, e2, e3, _)) in zip(axes, wyniki_mccv.items()):
        bp = ax.boxplot([e1, e2, e3], patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='poziom losowy (0.5)')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(f"Zbiór: {nazwa}", fontsize=11)
        ax.set_ylabel("Błąd klasyfikacji")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Adnotacje ze średnimi
        for i, err in enumerate([e1, e2, e3], 1):
            ax.text(i, err.mean() + 0.03, f"{err.mean():.3f}",
                    ha='center', fontsize=9, color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "plot_mccv_boxplot.png"), dpi=150)
    plt.show()

# ===========================================================================
# Częstość wyboru cech w MCCV (cechy stabilne)
# Pokazuje które cechy są dyskryminatywne
# ===========================================================================
def plot_feature_stability(wyniki_mccv, real_set="set1", top_n=30):
    """
    Barplot częstości wyboru cech w MCCV dla zbioru z sygnałem (set1).
    Odpowiada na punkt g) – indeksy cech dyskryminatywnych.
    """
    _, _, _, freq = wyniki_mccv[real_set]

    # Top N najczęściej wybieranych cech
    top_idx  = np.argsort(freq)[::-1][:top_n]
    top_freq = freq[top_idx]

    colors = ['#8B0000' if f >= STABILITY_THRESH else '#2196A6' for f in top_freq]

    plt.figure(figsize=(14, 5))
    bars = plt.bar(range(top_n), top_freq, color=colors, alpha=0.85)
    plt.axhline(STABILITY_THRESH, color='navy', linestyle='--', linewidth=1.5,
                label=f'próg stabilności ({STABILITY_THRESH*100:.0f}%)')
    plt.xticks(range(top_n), [f"C{top_idx[i]}" for i in range(top_n)], rotation=45, fontsize=8)
    plt.title(f"Częstość wyboru cech w MCCV – {real_set} (Scenariusz 3)\n", fontsize=11)
    plt.ylabel("Częstość wyboru")
    plt.xlabel("Indeks cechy")
    plt.ylim(0, 1.1)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"plot_feature_stability_{real_set}.png"), dpi=150)
    plt.show()

# ===========================================================================
# Porównanie średnich błędów scenariuszy (bar chart)
# Czytelna tabela wizualna do porównania zbiorów
# ===========================================================================
def plot_mean_errors_comparison(wyniki_mccv):
    """
    Grupowany wykres słupkowy średnich błędów MCCV.
    Ułatwia porównanie set1 vs set2 dla każdego scenariusza (punkt e, f).
    """
    scenariusze = ['Scen. 1\nResubstytucja', 'Scen. 2\nLeaky holdout', 'Scen. 3\nPoprawny holdout']
    x = np.arange(len(scenariusze))
    width = 0.35

    means_s1 = [wyniki_mccv["set1"][i].mean() for i in range(3)]
    stds_s1  = [wyniki_mccv["set1"][i].std()  for i in range(3)]
    means_s2 = [wyniki_mccv["set2"][i].mean() for i in range(3)]
    stds_s2  = [wyniki_mccv["set2"][i].std()  for i in range(3)]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, means_s1, width, yerr=stds_s1,
                   label='set1 (cechy dyskryminatywne)', color='#2196A6',
                   alpha=0.85, capsize=5)
    bars2 = ax.bar(x + width/2, means_s2, width, yerr=stds_s2,
                   label='set2 (zbiór losowy)', color='#E24B4A',
                   alpha=0.85, capsize=5)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='poziom losowy (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenariusze, fontsize=11)
    ax.set_ylabel("Średni błąd klasyfikacji (± std)")
    ax.set_title("Porównanie średnich błędów MCCV – set1 vs set2\n", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Wartości nad słupkami
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "plot_mean_errors_comparison.png"), dpi=150)
    plt.show()

def plot_mccv_boxplot_single(e1, e2, e3, title="ext_set"):
    """
    Boxplot rozkładu błędów MCCV dla jednego zbioru danych.
    Używany gdy nie ma drugiego zbioru do porównania (np. ext_set w zadaniu 2).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Rozkład błędów klasyfikacji – MCCV ({len(e1)} iteracji)\n"
                 f"Zbiór: {title}", fontsize=13, fontweight='bold')

    colors = ['#E24B4A', '#EF9F27', '#2196A6']
    labels = ['Scen. 1\nResubstytucja', 'Scen. 2\nLeaky holdout', 'Scen. 3\nPoprawny holdout']

    bp = ax.boxplot([e1, e2, e3], patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='poziom losowy (0.5)')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Błąd klasyfikacji")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    for i, err in enumerate([e1, e2, e3], 1):
        ax.text(i, err.mean() + 0.03, f"{err.mean():.3f}",
                ha='center', fontsize=9, color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"plot_mccv_boxplot_{title}.png"), dpi=150)
    plt.show()
