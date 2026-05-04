import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.decomposition import PCA
import scipy.io as sio

# Konfiguracja
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
plt.ioff()

# ============================================================
# Funkcje pomocnicze
# ============================================================
def komitet_glosowanie(klasyfikatory, X):
    predykcje = np.array([clf.predict(X) for clf in klasyfikatory])
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, predykcje)

def komitet_wazony(klasyfikatory, wagi, X):
    predykcje = np.array([clf.predict(X) for clf in klasyfikatory])
    wynik = np.zeros((X.shape[0], 2))
    for w, pred in zip(wagi, predykcje):
        wynik[np.arange(len(pred)), pred] += w
    return np.argmax(wynik, axis=1)

def rysuj_granice_2d(ax, X_2d, y, Z_values, title):
    """Rysuje granicę decyzyjną na danych 2D (już przetworzonych)"""
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = Z_values.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='bwr', edgecolor='k', s=25)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# ============================================================
# Funkcja ładująca dane z pliku .mat
# ============================================================
def wczytaj_dane_mat():
    """Próbuje wczytać dane z data7.mat lub dane7.mat"""
    for nazwa_pliku in ['data7.mat', 'dane7.mat']:
        if os.path.exists(nazwa_pliku):
            print(f"  Znaleziono plik: {nazwa_pliku}")
            try:
                mat = sio.loadmat(nazwa_pliku)
                
                if 'uczacy' in mat and 'testowy' in mat:
                    uczacy = mat['uczacy'][0, 0]
                    testowy = mat['testowy'][0, 0]
                    
                    X_train_mat = np.asarray(uczacy['X'], dtype=float)
                    y_train_mat = np.asarray(uczacy['D'], dtype=int).ravel()
                    X_test_mat = np.asarray(testowy['X'], dtype=float)
                    y_test_mat = np.asarray(testowy['D'], dtype=int).ravel()
                    
                    # Transpozycja jeśli trzeba
                    if X_train_mat.shape[0] > X_train_mat.shape[1]:
                        X_train_mat = X_train_mat.T
                    if X_test_mat.shape[0] > X_test_mat.shape[1]:
                        X_test_mat = X_test_mat.T
                    
                    X_all = np.vstack([X_train_mat, X_test_mat])
                    y_all = np.concatenate([y_train_mat, y_test_mat])
                    
                    print(f"  Wczytano: {X_all.shape[0]} próbek, {X_all.shape[1]} cech")
                    print(f"  Klasa 0: {np.sum(y_all==0)}, Klasa 1: {np.sum(y_all==1)}")
                    
                    # Sprawdź czy dane są 2D
                    if X_all.shape[1] == 2:
                        print(f"  Dane są 2D - będzie można rysować granice decyzyjne")
                    else:
                        print(f"  Dane mają {X_all.shape[1]} cech - granice będą na PCA(2D)")
                    
                    return X_all, y_all, nazwa_pliku
                    
            except Exception as e:
                print(f"  Błąd wczytywania {nazwa_pliku}: {e}")
    
    print("  Nie znaleziono plików .mat z danymi")
    return None, None, None

# ============================================================
# Główna funkcja analizy
# ============================================================
def analizuj_dane(X, y, prefix="synthetic"):
    """Pełna analiza: single, komitety, wykresy"""
    print(f"\n{'='*60}")
    print(f"Analiza danych: {prefix}")
    print(f"{'='*60}")
    print(f"  Wymiar danych: {X.shape[1]} cech")
    
    # Sprawdź wymiar i przygotuj PCA jeśli trzeba
    is_2d = (X.shape[1] == 2)
    if not is_2d:
        print(f"  Stosuję PCA do wizualizacji 2D...")
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        print(f"  Wyjaśniona wariancja: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        X_2d = X
    
    # Podział na zbiory
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Do wizualizacji 2D też dzielimy
    X_train_2d, X_test_2d, _, _ = train_test_split(
        X_2d, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # WYKRES 1: Dane wejściowe (2D lub PCA)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='bwr', edgecolor='k', s=30)
    title_suffix = '' if is_2d else ' (PCA)'
    ax.set_title(f'1. Dane: {prefix}{title_suffix}')
    ax.set_xlabel('x1' if is_2d else 'PC1')
    ax.set_ylabel('x2' if is_2d else 'PC2')
    fname = f'{prefix}_01_dane.png'
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {fname}")
    
    # Trenowanie singli
    classifiers = {
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Drzewo decyzyjne': DecisionTreeClassifier(max_depth=4, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, random_state=42),
        'Regresja logistyczna': LogisticRegression(max_iter=2000)
    }
    
    single_scores = {}
    single_models = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        single_scores[name] = accuracy_score(y_test, clf.predict(X_test))
        single_models[name] = clf
    
    # WYKRES 2: Słupkowe porównanie singli
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(single_scores.keys())
    values = list(single_scores.values())
    colors = ['steelblue', 'forestgreen', 'firebrick', 'darkorchid']
    bars = ax.bar(names, values, color=colors)
    ax.set_title(f'2. Jakość pojedynczych klasyfikatorów ({prefix})')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, 1.0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    fig.tight_layout()
    fname = f'{prefix}_02_single_scores.png'
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  ✓ {fname}")
    
    # WYKRES 3: Granice singli (tylko jeśli 2D lub przez PCA)
    if is_2d:
        # Dane oryginalne 2D - normalne granice
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        axes = axes.ravel()
        
        for i, (name, clf) in enumerate(classifiers.items()):
            h = 0.02
            x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
            y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            
            axes[i].contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
            axes[i].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolor='k', s=25)
            axes[i].set_title(f'{name}\nAcc={single_scores[name]:.3f}', fontsize=9)
            axes[i].set_xticks([]); axes[i].set_yticks([])
        
        fig.suptitle(f'3. Granice decyzyjne - single ({prefix})', fontsize=12)
        fig.tight_layout()
        fname = f'{prefix}_03_granice_single.png'
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
        plt.close(fig)
        print(f"  ✓ {fname}")
    else:
        # Dane wielowymiarowe - PCA + kontur przez predykcję na siatce PCA
        print(f"  Pomijam granice dla danych wielowymiarowych (zapisuję tylko PCA scatter)")
        
        # Alternatywnie: granice na PCA (trenuj modele na PCA)
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        axes = axes.ravel()
        
        # Trenuj modele na danych PCA do wizualizacji
        pca_models = {}
        for name, clf_type in [('KNN (k=5)', KNeighborsClassifier(n_neighbors=5)),
                                ('Drzewo', DecisionTreeClassifier(max_depth=4, random_state=42)),
                                ('SVM (RBF)', SVC(kernel='rbf', C=1.0, random_state=42)),
                                ('RegLog', LogisticRegression(max_iter=2000))]:
            m = clone(clf_type)
            m.fit(X_train_2d, y_train)
            pca_models[name] = m
        
        for i, (name, clf) in enumerate(pca_models.items()):
            h = 0.02
            x_min, x_max = X_test_2d[:, 0].min() - 0.5, X_test_2d[:, 0].max() + 0.5
            y_min, y_max = X_test_2d[:, 1].min() - 0.5, X_test_2d[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            
            axes[i].contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
            axes[i].scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='bwr', edgecolor='k', s=25)
            axes[i].set_title(f'{name} (na PCA)\nAcc={single_scores[name.replace(" (k=5)", "").replace(" (RBF)", "")] if name in single_scores else accuracy_score(y_test, clf.predict(X_test_2d)):.3f}', fontsize=9)
            axes[i].set_xticks([]); axes[i].set_yticks([])
        
        fig.suptitle(f'3. Granice decyzyjne na PCA - single ({prefix})', fontsize=12)
        fig.tight_layout()
        fname = f'{prefix}_03_granice_single_pca.png'
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
        plt.close(fig)
        print(f"  ✓ {fname}")
    
    # ============================================================
    # Budowanie komitetów (na oryginalnych danych!)
    # ============================================================
    
    # --- Bagging ---
    print("  Budowanie komitetów Bagging...")
    wyniki_bagging = {'N': [], 'majority': [], 'weighted': []}
    bagging_komitety = {}
    
    for N in [3, 5, 7]:
        klf = []; wagi = []
        for i in range(N):
            idx = np.random.choice(len(X_train), len(X_train), replace=True)
            dt = DecisionTreeClassifier(max_depth=4, random_state=i)
            dt.fit(X_train[idx], y_train[idx])
            klf.append(dt)
            wagi.append(accuracy_score(y_test, dt.predict(X_test)))
        
        maj_acc = accuracy_score(y_test, komitet_glosowanie(klf, X_test))
        waz_acc = accuracy_score(y_test, komitet_wazony(klf, wagi, X_test))
        
        wyniki_bagging['N'].append(N)
        wyniki_bagging['majority'].append(maj_acc)
        wyniki_bagging['weighted'].append(waz_acc)
        bagging_komitety[N] = (klf, wagi)
        print(f"    N={N}: majority={maj_acc:.3f}, weighted={waz_acc:.3f}")
    
    # --- Różne klasyfikatory ---
    print("  Budowanie komitetów różnorodnych...")
    wyniki_diverse = {'N': [], 'majority': [], 'weighted': []}
    diverse_komitety = {}
    
    modele = [
        KNeighborsClassifier(n_neighbors=3),
        KNeighborsClassifier(n_neighbors=7),
        DecisionTreeClassifier(max_depth=3, random_state=1),
        DecisionTreeClassifier(max_depth=6, random_state=2),
        SVC(kernel='rbf', C=0.5, random_state=3),
        SVC(kernel='rbf', C=2.0, random_state=4),
        LogisticRegression(max_iter=2000)
    ]
    
    for N in [3, 5, 7]:
        klf = []; wagi = []
        for i in range(N):
            m = clone(modele[i]); m.fit(X_train, y_train)
            klf.append(m)
            wagi.append(accuracy_score(y_test, m.predict(X_test)))
        
        maj_acc = accuracy_score(y_test, komitet_glosowanie(klf, X_test))
        waz_acc = accuracy_score(y_test, komitet_wazony(klf, wagi, X_test))
        
        wyniki_diverse['N'].append(N)
        wyniki_diverse['majority'].append(maj_acc)
        wyniki_diverse['weighted'].append(waz_acc)
        diverse_komitety[N] = (klf, wagi)
        print(f"    N={N}: majority={maj_acc:.3f}, weighted={waz_acc:.3f}")
    
    # --- Różne parametry KNN ---
    print("  Budowanie komitetów KNN (różne k)...")
    wyniki_knn = {'N': [], 'majority': [], 'weighted': []}
    knn_komitety = {}
    
    for N in [3, 5, 7]:
        klf = []; wagi = []
        parametry_k = np.linspace(1, 15, N, dtype=int)
        for k in parametry_k:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            klf.append(knn)
            wagi.append(accuracy_score(y_test, knn.predict(X_test)))
        
        maj_acc = accuracy_score(y_test, komitet_glosowanie(klf, X_test))
        waz_acc = accuracy_score(y_test, komitet_wazony(klf, wagi, X_test))
        
        wyniki_knn['N'].append(N)
        wyniki_knn['majority'].append(maj_acc)
        wyniki_knn['weighted'].append(waz_acc)
        knn_komitety[N] = (klf, wagi)
        print(f"    N={N}: majority={maj_acc:.3f}, weighted={waz_acc:.3f}")
    
    # ============================================================
    # WYKRES 4: Trend N
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(wyniki_bagging['N'], wyniki_bagging['majority'], 'b-o', label='Bagging - majority')
    ax1.plot(wyniki_bagging['N'], wyniki_bagging['weighted'], 'b--o', label='Bagging - weighted')
    ax1.plot(wyniki_diverse['N'], wyniki_diverse['majority'], 'g-s', label='Różne - majority')
    ax1.plot(wyniki_diverse['N'], wyniki_diverse['weighted'], 'g--s', label='Różne - weighted')
    ax1.plot(wyniki_knn['N'], wyniki_knn['majority'], 'r-^', label='KNN param - majority')
    ax1.plot(wyniki_knn['N'], wyniki_knn['weighted'], 'r--^', label='KNN param - weighted')
    
    best_single = max(single_scores.values())
    ax1.axhline(best_single, color='gray', linestyle=':', linewidth=2,
                label=f'Najlepszy singiel ({best_single:.3f})')
    ax1.set_xlabel('Liczba klasyfikatorów (N)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'5. Wpływ liczby klasyfikatorów ({prefix})')
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([3, 5, 7])
    
    kategorie = ['Najlepszy\nsingiel', 'Bagging\nN=7', 'Różne\nN=7', 'KNN param\nN=7']
    wartosci = [best_single,
                max(wyniki_bagging['weighted']), 
                max(wyniki_diverse['weighted']), 
                max(wyniki_knn['weighted'])]
    bar_colors = ['gray', 'steelblue', 'forestgreen', 'firebrick']
    bars = ax2.bar(kategorie, wartosci, color=bar_colors, edgecolor='black', linewidth=1.2)
    ax2.set_title(f'Najlepsze komitety vs singiel ({prefix})')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0.4, 1.0)
    for bar, val in zip(bars, wartosci):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    fname = f'{prefix}_04_trend_N.png'
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"  ✓ {fname}")
    
    # WYKRES 5 tylko dla danych 2D
    if is_2d:
        fig, axes = plt.subplots(3, 2, figsize=(12, 14))
        
        typy = [
            ('Bagging (różne dane)', bagging_komitety[7]),
            ('Różne klasyfikatory', diverse_komitety[7]),
            ('KNN różne k', knn_komitety[7])
        ]
        
        for idx, (nazwa, (klf, wagi)) in enumerate(typy):
            maj_acc = accuracy_score(y_test, komitet_glosowanie(klf, X_test))
            waz_acc = accuracy_score(y_test, komitet_wazony(klf, wagi, X_test))
            
            # Rysuj granice dla majority
            h = 0.02
            x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
            y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            
            Z_maj = komitet_glosowanie(klf, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            axes[idx, 0].contourf(xx, yy, Z_maj, alpha=0.4, cmap='bwr')
            axes[idx, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolor='k', s=25)
            axes[idx, 0].set_title(f'{nazwa}\nMajority (Acc={maj_acc:.3f})', fontsize=9)
            axes[idx, 0].set_xticks([]); axes[idx, 0].set_yticks([])
            
            Z_waz = komitet_wazony(klf, wagi, np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            axes[idx, 1].contourf(xx, yy, Z_waz, alpha=0.4, cmap='bwr')
            axes[idx, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolor='k', s=25)
            axes[idx, 1].set_title(f'{nazwa}\nWeighted (Acc={waz_acc:.3f})', fontsize=9)
            axes[idx, 1].set_xticks([]); axes[idx, 1].set_yticks([])
        
        fig.suptitle(f'6. Granice decyzyjne komitetów N=7 ({prefix})', fontsize=14)
        fig.tight_layout()
        fname = f'{prefix}_05_granice_komitety.png'
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
        plt.close(fig)
        print(f"  ✓ {fname}")
    else:
        print(f"  Pomijam granice komitetów dla danych >2D (są na wykresie trendu)")
    
    return single_scores, wyniki_bagging, wyniki_diverse, wyniki_knn

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("ANALIZA KOMITETÓW KLASYFIKATORÓW")
    print("="*60)
    
    # --- 1. Dane syntetyczne ---
    print("\n🔧 Generowanie danych syntetycznych...")
    X_syn, y_syn = make_moons(n_samples=500, noise=0.3, random_state=42)
    wyniki_syn = analizuj_dane(X_syn, y_syn, prefix="synthetic")
    
    # --- 2. Dane z pliku .mat (jeśli istnieje) ---
    # print("\n🔧 Sprawdzanie plików .mat...")
    # X_mat, y_mat, nazwa_pliku = wczytaj_dane_mat()
    
    # wyniki_mat = None
    # if X_mat is not None:
    #     prefix_mat = nazwa_pliku.replace('.mat', '')
    #     wyniki_mat = analizuj_dane(X_mat, y_mat, prefix=prefix_mat)
    
    # # --- Podsumowanie końcowe ---
    # print("\n" + "="*60)
    # print("PODSUMOWANIE KOŃCOWE")
    # print("="*60)
    
    # for prefix, wyniki in [("synthetic", wyniki_syn)] + \
    #                       ([("MAT", wyniki_mat)] if wyniki_mat else []):
    #     single_scores, bag, div, knn = wyniki
    #     best_single_name = max(single_scores, key=single_scores.get)
    #     best_single_acc = single_scores[best_single_name]
        
    #     print(f"\n📊 Dane: {prefix}")
    #     print(f"  Wymiar danych: {X_syn.shape[1] if prefix=='synthetic' else X_mat.shape[1]} cech")
    #     print(f"  Najlepszy singiel: {best_single_name} ({best_single_acc:.3f})")
    #     print(f"  Bagging N=7 weighted:    {max(bag['weighted']):.3f}")
    #     print(f"  Różne N=7 weighted:      {max(div['weighted']):.3f}")
    #     print(f"  KNN param N=7 weighted:  {max(knn['weighted']):.3f}")
        
    #     # Sprawdź czy komitet jest lepszy
    #     best_committee = max(max(bag['weighted']), max(div['weighted']), max(knn['weighted']))
    #     if best_committee > best_single_acc:
    #         print(f"  ✅ Komitet poprawia wynik o +{best_committee - best_single_acc:.3f}")
    #     else:
    #         print(f"  ⚠️ Komitet NIE poprawił wyniku (singiel lepszy o {best_single_acc - best_committee:.3f})")
    
    # print(f"\n✅ Wszystkie wykresy zapisane w folderze: {PLOTS_DIR}/")
    