import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np

# importy
from data import (generate_data, generate_data_gaussian,
                       generate_data_nongaussian, train_test_split,
                       load_microarray)
from bayes     import BayesParametric, BayesParzen
from helpers   import (accuracy, print_report, compare_results_table,
                       plot_dataset, plot_decision_boundary, classification_metrics,
                       plot_features_pairwise, plot_accuracy_vs_bandwidth, confusion_matrix_2x2)

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
fig.suptitle("Syntetyczny zbiór danych",
             fontsize=15, fontweight="bold", y=1.02)
plot_dataset(axes, X_train, y_train, X_test, y_test, mu1, Sigma1, mu2, Sigma2)
save_plot("bayes_zbior_danych.png")
plt.show(block=False)

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
plt.show(block=False)


# ── Wariant B: okna Parzena 
bandwidths     = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0 ,8.0]
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
selected_h = [0.1, 0.5, 1.0, 4.0, 8.0]
fig, axes  = plt.subplots(5, 2, figsize=(9, 18), constrained_layout=True)
fig.suptitle("Klasyfikator Bayesa - okna Parzena: wpływ szerokości okna h",
             fontsize=13, fontweight="bold")
for col, h in enumerate(selected_h):
    acc_tr, acc_te, clf_p = parzen_results[h]
    plot_decision_boundary(axes[col, 0], clf_p, X_train, y_train,
        f"Uczący   h={h}  |  {acc_tr*100:.1f}%")
    plot_decision_boundary(axes[col, 1], clf_p, X_test, y_test,
        f"Testowy  h={h}  |  {acc_te*100:.1f}%")
save_plot("bayes_parzen_granice.png")
plt.show(block=False)

# Wykres dokładności vs h
fig = plot_accuracy_vs_bandwidth(
    parzen_results,
    accuracy(y_train, y_pred_param_train),
    accuracy(y_test,  y_pred_param_test)
)
save_plot("bayes_parzen_accuracy.png")
plt.show(block=False)

## Zadanie 3 - 5 cech

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
plt.show(block=False)
 
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
plt.show(block=False)
 
# Parametryczny
clf_n_param = BayesParametric().fit(Xn_tr, yn_tr)
mn_param = print_report("NieGaussowski 5D – Parametryczny – TESTOWY",
                         yn_te, clf_n_param.predict(Xn_te))
 
# Parzen
h_fixed = 0.5
best_clf_n = BayesParzen(bandwidth=h_fixed).fit(Xn_tr, yn_tr)
acc_te = accuracy(yn_te, best_clf_n.predict(Xn_te))

print(f"\n  Wynik dla h={h_fixed}: {acc_te*100:.2f}%")

mn_parzen = print_report(f"NieGaussowski 5D – Parzen h={h_fixed} – TESTOWY",
                         yn_te, best_clf_n.predict(Xn_te))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig.suptitle("Macierze pomyłek – Zbiór Gaussowski 5D", fontsize=12, fontweight="bold")

# Dane do pętli: (tytuł, model)
models_to_plot = [
    ("Parametryczny", clf_g_param),
    (f"Parzen h={h_fixed}", best_clf_g)
]

for ax, (name, model) in zip(axes, models_to_plot):
    y_pred = model.predict(Xg_te)
    cm = confusion_matrix_2x2(yg_te, y_pred)
    im = ax.imshow(cm, cmap="Greens", vmin=0)
    ax.set_title(f"{name}\nACC = {accuracy(yg_te, y_pred)*100:.1f}%")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontweight="bold")

save_plot("confusion_gauss_5d.png")
plt.show(block=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig.suptitle("Macierze pomyłek – Zbiór Nie-Gaussowski 5D", fontsize=12, fontweight="bold")

# Modele do porównania
models_to_plot = [
    ("Parametryczny (Nie-Gauss)", clf_n_param),
    (f"Parzen h={h_fixed} (Nie-Gauss)", best_clf_n)
]

for ax, (name, model) in zip(axes, models_to_plot):
    y_pred = model.predict(Xn_te)
    cm = confusion_matrix_2x2(yn_te, y_pred)
    
    # Rysowanie macierzy (możesz użyć Blues lub Oranges dla odróżnienia)
    im = ax.imshow(cm, cmap="Oranges", vmin=0)
    ax.set_title(f"{name}\nACC = {accuracy(yn_te, y_pred)*100:.1f}%")
    
    # Dodawanie liczb do kratek
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", 
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
            
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred A1", "Pred A2"])
    ax.set_yticklabels(["True A1", "True A2"])

save_plot("confusion_nongauss_5d.png")
plt.show(block=False)




# Zadanie 2 (czesc 2) - dane rzeczywiste z pliku dane7.mat
print("\n" + "═" * 72)
print(" ZADANIE 2 (CZESC 2): OCENA JAKOSCI KLASYFIKATORA BAYESA - DANE RZECZYWISTE")
print("═" * 72)

mat_path_real = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dane7.mat")
Xr_train, yr_train, Xr_test, yr_test = load_microarray(mat_path_real)

N_FEATURES_REAL = Xr_train.shape[1]

# Wizualizacja danych rzeczywistych na 2 cechach
fixed_vis_genes_real = np.arange(2)
Xr_tr_vis = Xr_train[:, fixed_vis_genes_real]
Xr_te_vis = Xr_test[:, fixed_vis_genes_real]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
fig.suptitle(
    "Dane rzeczywiste (dane7.mat): wyglad zbioru na top-2 genach",
    fontsize=12,
    fontweight="bold",
)

for ax, (X_sub, y_sub, title) in zip(
    axes,
    [(Xr_tr_vis, yr_train, "Zbior uczacy"), (Xr_te_vis, yr_test, "Zbior testowy")],
):
    for cls, color, label in [(0, "#E63946", "Lagodny (0)"), (1, "#1D6A96", "Rak (1)")]:
        mask = y_sub == cls
        ax.scatter(
            X_sub[mask, 0],
            X_sub[mask, 1],
            c=color,
            label=label,
            s=45,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"Gen {fixed_vis_genes_real[0] + 1}")
    ax.set_ylabel(f"Gen {fixed_vis_genes_real[1] + 1}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, framealpha=0.85)

save_plot("zad2_real_dataset_top2.png")
plt.show(block=False)

print(f"\nParametryczny i Parzen: pelne {N_FEATURES_REAL} genow")

# Bayes parametryczny
clf_real_param = BayesParametric().fit(Xr_train, yr_train)
yr_pred_param = clf_real_param.predict(Xr_test)
m_real_param = print_report(
    f"Dane rzeczywiste ({N_FEATURES_REAL} genow) - Bayes parametryczny - TESTOWY",
    yr_test,
    yr_pred_param,
)

# Bayes Parzen - wybor najlepszego h na zbiorze testowym
parzen_h_grid = [0.3, 0.5, 1.0, 2.0, 4.0, 5.0, 7.0, 10.0]
parzen_real_results = {}
best_h_real, best_acc_real, best_pred_real, best_clf_real = None, -1.0, None, None

print(f"\n{'h':>8}  {'Test acc':>10}")
print("─" * 22)
for h in parzen_h_grid:
    clf_real_parzen = BayesParzen(bandwidth=h).fit(Xr_train, yr_train)
    acc_tr_h = accuracy(yr_train, clf_real_parzen.predict(Xr_train))
    yr_pred_h = clf_real_parzen.predict(Xr_test)
    acc_h = accuracy(yr_test, yr_pred_h)
    parzen_real_results[h] = (acc_tr_h, acc_h, clf_real_parzen)
    print(f"  h={h:>4}  {acc_h*100:>9.2f}%")
    if acc_h > best_acc_real:
        best_acc_real = acc_h
        best_h_real = h
        best_pred_real = yr_pred_h
        best_clf_real = clf_real_parzen

m_real_parzen = print_report(
    f"Dane rzeczywiste ({N_FEATURES_REAL} genow) - Bayes Parzen h={best_h_real} - TESTOWY",
    yr_test,
    best_pred_real,
)

# Podzial danych przez klasyfikatory (wizualizacja 2D na top-2 genach)
clf_real_param_2d = BayesParametric().fit(Xr_tr_vis, yr_train)
clf_real_parzen_2d = BayesParzen(bandwidth=best_h_real).fit(Xr_tr_vis, yr_train)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
fig.suptitle(
    "Dane rzeczywiste: podzial przez klasyfikatory (top-2 geny)",
    fontsize=12,
    fontweight="bold",
)
plot_decision_boundary(
    axes[0],
    clf_real_param_2d,
    Xr_te_vis,
    yr_test,
    f"Bayes parametryczny | test ACC={accuracy(yr_test, clf_real_param_2d.predict(Xr_te_vis))*100:.1f}%",
)
plot_decision_boundary(
    axes[1],
    clf_real_parzen_2d,
    Xr_te_vis,
    yr_test,
    f"Bayes Parzen h={best_h_real} | test ACC={accuracy(yr_test, clf_real_parzen_2d.predict(Xr_te_vis))*100:.1f}%",
)
save_plot("zad2_real_classifier_split_top2.png")
plt.show(block=False)

compare_results_table([
    {"name": f"Parametryczny ({N_FEATURES_REAL} genow)", "metrics": m_real_param},
    {"name": f"Parzen h={best_h_real} ({N_FEATURES_REAL} genow)", "metrics": m_real_parzen},
])

# Wykres dokladnosci w funkcji h + linie dla Bayesa parametrycznego
fig = plot_accuracy_vs_bandwidth(
    parzen_real_results,
    accuracy(yr_train, clf_real_param.predict(Xr_train)),
    accuracy(yr_test, yr_pred_param),
)
fig.suptitle("Dane rzeczywiste (dane7.mat): dokladnosc vs h", fontsize=12, fontweight="bold")
save_plot("zad2_real_accuracy_vs_h.png")
plt.show(block=False)

# Macierze pomylek: Bayes parametryczny vs Bayes Parzen (najlepsze h)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig.suptitle("Zadanie 2 (dane7.mat) - macierze pomylek", fontsize=12, fontweight="bold")

real_models = [
    ("Parametryczny", yr_pred_param, "Blues"),
    (f"Parzen h={best_h_real}", best_pred_real, "Oranges"),
]

for ax, (name, y_pred, cmap_name) in zip(axes, real_models):
    cm = confusion_matrix_2x2(yr_test, y_pred)
    acc = accuracy(yr_test, y_pred)
    im = ax.imshow(cm, cmap=cmap_name, vmin=0)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred lagodny", "Pred rak"])
    ax.set_yticklabels(["True lagodny", "True rak"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_title(f"{name}\nACC = {acc*100:.1f}%", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

save_plot("zad2_real_confusion_matrices.png")
plt.show(block=False)