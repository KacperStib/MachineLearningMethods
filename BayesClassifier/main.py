import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# importy
from data import (generate_data, generate_data_gaussian,
                       generate_data_nongaussian, train_test_split,
                       load_microarray, select_features_fisher)
from bayes     import BayesParametric, BayesParzen
from helpers   import (accuracy, print_report, compare_results_table,
                       plot_dataset, plot_decision_boundary, classification_metrics,
                       plot_features_pairwise, plot_accuracy_vs_bandwidth)

output_dir = "img"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Utworzono folder: {output_dir}")

def save_plot(filename):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Zapisano: {path}")

# Zadanie 1
X, y, mu1, Sigma1, mu2, Sigma2 = generate_data(seed=42)
X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=100)

print(f"Zbiór uczący  - A1: {(y_train==0).sum():3d}, A2: {(y_train==1).sum():3d}")
print(f"Zbiór testowy - A1: {(y_test ==0).sum():3d}, A2: {(y_test ==1).sum():3d}")

# ── Wykres 1: zbiory danych ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
fig.suptitle("Syntetyczny zbiór danych - klasyfikator Bayesa",
             fontsize=15, fontweight="bold", y=1.02)
plot_dataset(axes, X_train, y_train, X_test, y_test, mu1, Sigma1, mu2, Sigma2)
save_plot("bayes_zbior_danych.png")
plt.show()

# Zadanie 2
# ── Wariant A: parametryczny 
clf_param = BayesParametric().fit(X_train, y_train)
y_pred_param_train = clf_param.predict(X_train)
y_pred_param_test  = clf_param.predict(X_test)

print("\n" + "═"*50)
print(" WYNIKI - KLASYFIKATOR PARAMETRYCZNY")
print("═"*50)
print_report("Parametryczny - zbiór UCZĄCY",  y_train, y_pred_param_train)
print_report("Parametryczny - zbiór TESTOWY", y_test,  y_pred_param_test)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
fig.suptitle("Klasyfikator Bayesa - wariant parametryczny (rozkład normalny)",
             fontsize=13, fontweight="bold")
plot_decision_boundary(axes[0], clf_param, X_train, y_train,
    f"Zbiór uczący  |  acc = {accuracy(y_train, y_pred_param_train)*100:.1f}%")
plot_decision_boundary(axes[1], clf_param, X_test, y_test,
    f"Zbiór testowy  |  acc = {accuracy(y_test, y_pred_param_test)*100:.1f}%")
save_plot("bayes_param.png")
plt.show()


# ── Wariant B: okna Parzena 
bandwidths     = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0]
parzen_results = {}

print("\n\n" + "═"*50)
print(" WYNIKI - KLASYFIKATOR PARZEN")
print("═"*50)
print(f"\n{'h':>8}  {'Train acc':>10}  {'Test acc':>10}")
print("─"*34)
for h in bandwidths:
    clf_p  = BayesParzen(bandwidth=h).fit(X_train, y_train)
    acc_tr = accuracy(y_train, clf_p.predict(X_train))
    acc_te = accuracy(y_test,  clf_p.predict(X_test))
    parzen_results[h] = (acc_tr, acc_te, clf_p)
    print(f"  h={h:>4}  {acc_tr*100:>9.2f}%  {acc_te*100:>9.2f}%")

# Granice decyzyjne dla wybranych h
selected_h = [0.1, 0.5, 1.0, 4.0]
fig, axes  = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
fig.suptitle("Klasyfikator Bayesa - okna Parzena: wpływ szerokości okna h",
             fontsize=13, fontweight="bold")
for col, h in enumerate(selected_h):
    acc_tr, acc_te, clf_p = parzen_results[h]
    plot_decision_boundary(axes[0, col], clf_p, X_train, y_train,
        f"Uczący   h={h}  |  {acc_tr*100:.1f}%")
    plot_decision_boundary(axes[1, col], clf_p, X_test, y_test,
        f"Testowy  h={h}  |  {acc_te*100:.1f}%")
save_plot("bayes_parzen_granice.png")
plt.show()

# Wykres dokładności vs h
fig = plot_accuracy_vs_bandwidth(
    parzen_results,
    accuracy(y_train, y_pred_param_train),
    accuracy(y_test,  y_pred_param_test)
)
save_plot("bayes_parzen_accuracy.png")
plt.show()

# Zadanie 3 - 5 cech
# Zbior Gaussowski
print("\n\n" + "="*52)
print(" ZBIOR 5-CECHOWY – GAUSSOWSKI (rozklady normalne)")
print("="*52)
 
Xg, yg = generate_data_gaussian(n_features=5, N=100, sep=2.0, seed=42)
Xg_tr, yg_tr, Xg_te, yg_te = train_test_split(Xg, yg, train_size=100)
 
# Wizualizacja – macierz par cech
fig = plot_features_pairwise(Xg, yg, n_show=5,
    title="Zbior gaussowski – 5 cech (diagonal: histogram, reszta: scatter)")
save_plot("gaussian_5_cech.png")
plt.show()
 
# Parametryczny
clf_g_param = BayesParametric().fit(Xg_tr, yg_tr)
mg_param = print_report("Gaussian 5D – Parametryczny – TESTOWY",
                         yg_te, clf_g_param.predict(Xg_te))
 
# Parzen (kilka h)
best_h_g, best_acc_g, best_clf_g = None, -1, None
print(f"\n  {'h':>6}  {'Test acc':>10}")
print("  " + "-"*20)
for h in [0.3, 0.5, 1.0, 2.0]:
    clf_gp = BayesParzen(bandwidth=h).fit(Xg_tr, yg_tr)
    acc_te = accuracy(yg_te, clf_gp.predict(Xg_te))
    print(f"  h={h:>4}  {acc_te*100:>9.2f}%")
    if acc_te > best_acc_g:
        best_acc_g, best_h_g, best_clf_g = acc_te, h, clf_gp
 
mg_parzen = print_report(f"Gaussian 5D – Parzen h={best_h_g} (najlepsze) – TESTOWY",
                          yg_te, best_clf_g.predict(Xg_te))

# Zbior nieGaussowski
print("\n\n" + "="*52)
print(" ZBIOR 5-CECHOWY – NIEgaussowski (mieszanina + jednostajny)")
print("="*52)
 
Xn, yn = generate_data_nongaussian(n_features=5, N=100, seed=42)
Xn_tr, yn_tr, Xn_te, yn_te = train_test_split(Xn, yn, train_size=100)
 
# Wizualizacja
fig = plot_features_pairwise(Xn, yn, n_show=5,
    title="Zbior nieGaussowski – 5 cech (diagonal: histogram, reszta: scatter)")
save_plot("nongaussian_5_cech.png")
plt.show()
 
# Parametryczny
clf_n_param = BayesParametric().fit(Xn_tr, yn_tr)
mn_param = print_report("NieGaussowski 5D – Parametryczny – TESTOWY",
                         yn_te, clf_n_param.predict(Xn_te))
 
# Parzen (kilka h)
best_h_n, best_acc_n, best_clf_n = None, -1, None
print(f"\n  {'h':>6}  {'Test acc':>10}")
print("  " + "-"*20)
for h in [0.3, 0.5, 1.0, 2.0]:
    clf_np = BayesParzen(bandwidth=h).fit(Xn_tr, yn_tr)
    acc_te = accuracy(yn_te, clf_np.predict(Xn_te))
    print(f"  h={h:>4}  {acc_te*100:>9.2f}%")
    if acc_te > best_acc_n:
        best_acc_n, best_h_n, best_clf_n = acc_te, h, clf_np
 
mn_parzen = print_report(f"NieGaussowski 5D – Parzen h={best_h_n} (najlepsze) – TESTOWY",
                          yn_te, best_clf_n.predict(Xn_te))
 
# ZADANIE 4 - realne dane
# ── sciezka do pliku 
MAT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dane7.mat")
 
print("=" * 60)
print(" DANE RZECZYWISTE – MIKROMACIERZE DNA")
print(" Rak tarczycy: 0=lagodny, 1=rak brodawkowaty")
print("=" * 60 + "\n")
 
X_train, y_train, X_test, y_test = load_microarray(MAT_PATH)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1. Wybor cech wskaznikiem Fishera – top-N genow
 
N_FEATS_LIST  = [5, 10, 20, 50]
BANDWIDTHS    = [0.3, 0.5, 1.0, 2.0, 4.0]
 
summary = []   # do tabeli zbiorczej i wykresu
 
for N_FEAT in N_FEATS_LIST:
    X_tr, X_te, top_idx, f_scores = select_features_fisher(
        X_train, y_train, X_test, n_features=N_FEAT)
 
    print(f"\n{'═'*60}")
    print(f" Top-{N_FEAT} genow (wskaznik Fishera)")
    print(f" Indeksy genow: {top_idx[:5].tolist()} ...")
    print(f"{'═'*60}")
 
    # -- Parametryczny --
    clf_p = BayesParametric().fit(X_tr, y_train)
    m_p   = print_report(f"Parametryczny – top-{N_FEAT} genow", y_test, clf_p.predict(X_te))
 
    # -- Parzen: wszystkie h, zapisz najlepsze --
    print(f"\n  Parzen – przeglad h:")
    print(f"  {'h':>6}  {'Train acc':>10}  {'Test acc':>10}")
    print(f"  {'─'*30}")
    best_h, best_acc, best_m = None, -1, None
    for h in BANDWIDTHS:
        clf_z  = BayesParzen(bandwidth=h).fit(X_tr, y_train)
        acc_tr = accuracy(y_train, clf_z.predict(X_tr))
        acc_te = accuracy(y_test,  clf_z.predict(X_te))
        marker = " <-- najlepsze" if acc_te > best_acc else ""
        print(f"  h={h:>4}  {acc_tr*100:>9.2f}%  {acc_te*100:>9.2f}%{marker}")
        if acc_te > best_acc:
            best_acc = acc_te
            best_h   = h
            best_m   = classification_metrics(y_test, clf_z.predict(X_te))
 
    print()
    print_report(f"Parzen h={best_h} (najlepsze) – top-{N_FEAT} genow",
                 y_test, BayesParzen(bandwidth=best_h).fit(X_tr, y_train).predict(X_te))
 
    summary.append({
        "n":         N_FEAT,
        "acc_param": m_p['ACC'],
        "acc_parzen": best_acc,
        "best_h":    best_h,
        "m_param":   m_p,
        "m_parzen":  best_m,
    })
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. Tabela zbiorcza
 
compare_results_table(
    [{"name": f"Param     top-{r['n']:>2} genow", "metrics": r['m_param']}  for r in summary] +
    [{"name": f"Parzen h={r['best_h']} top-{r['n']:>2} genow", "metrics": r['m_parzen']} for r in summary]
)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. Wykresy
 
ns          = [r['n']           for r in summary]
accs_param  = [r['acc_param'] * 100  for r in summary]
accs_parzen = [r['acc_parzen'] * 100 for r in summary]
 
# --- Wykres 1: dokladnosc vs liczba genow ---
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
ax.plot(ns, accs_param,  "o-",  color="#E63946", linewidth=2,
        markersize=9, label="Parametryczny")
ax.plot(ns, accs_parzen, "s--", color="#1D6A96", linewidth=2,
        markersize=9, label=f"Parzen (najlepsze h)")
for i, r in enumerate(summary):
    ax.annotate(f"h={r['best_h']}", (ns[i], accs_parzen[i]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
                color="#1D6A96")
ax.set_xlabel("Liczba wybranych genow (top Fishera)", fontsize=11)
ax.set_ylabel("Dokladnosc na zbiorze testowym [%]",   fontsize=11)
ax.set_title("Mikromacierze DNA – dokladnosc vs liczba genow", fontsize=12, fontweight="bold")
ax.set_xticks(ns)
ax.set_ylim(40, 108)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=10, framealpha=0.9)
save_plot("microarray_acc_vs_genes.png")
plt.show()
 
 
# --- Wykres 2: macierze pomylek dla najlepszego N (top-10) ---
best_n = summary[1]   # top-10
X_tr10, X_te10, _, _ = select_features_fisher(X_train, y_train, X_test, n_features=10)
 
clf_p10 = BayesParametric().fit(X_tr10, y_train)
clf_z10 = BayesParzen(bandwidth=best_n['best_h']).fit(X_tr10, y_train)
 
preds = {
    "Parametryczny\n(top-10 genow)":          clf_p10.predict(X_te10),
    f"Parzen h={best_n['best_h']}\n(top-10 genow)": clf_z10.predict(X_te10),
}
 
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig.suptitle("Macierze pomylek – zbior testowy (mikromacierze DNA)",
             fontsize=12, fontweight="bold")
 
for ax, (title, y_pred) in zip(axes, preds.items()):
    from helpers import confusion_matrix_2x2
    cm  = confusion_matrix_2x2(y_test, y_pred)
    acc = accuracy(y_test, y_pred)
    im  = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred lagodny\n(0)", "Pred rak\n(1)"])
    ax.set_yticklabels(["True lagodny (0)", "True rak (1)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_title(f"{title}\nACC = {acc*100:.1f}%", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
 
save_plot("microarray_confusion.png")
plt.show()
 
# --- Wykres 3: rozklady top-2 genow dla obu klas ---
X_tr2, X_te2, top2_idx, _ = select_features_fisher(
    X_train, y_train, X_test, n_features=2)
 
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
fig.suptitle("Rozklady ekspresji 2 najlepszych genow (zbior uczacy)",
             fontsize=12, fontweight="bold")
COLORS = {0: "#E63946", 1: "#1D6A96"}
LABELS = {0: "Lagodny (0)", 1: "Rak (1)"}
for i, ax in enumerate(axes):
    for cls in [0, 1]:
        mask = y_train == cls
        ax.hist(X_tr2[mask, i], bins=20, color=COLORS[cls],
                alpha=0.6, density=True, label=LABELS[cls], edgecolor="white")
    ax.set_title(f"Gen nr {top2_idx[i]+1} (indeks w macierzy)", fontsize=10)
    ax.set_xlabel("Poziom ekspresji")
    ax.set_ylabel("Gestosc")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
 
save_plot("microarray_top2_genes.png")
plt.show()