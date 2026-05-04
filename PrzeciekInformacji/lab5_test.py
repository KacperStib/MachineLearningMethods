import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss

# ===========================================================================
# PARAMETRY
# ===========================================================================
SET1_FILE        = "S6_set1.csv"
SET2_FILE        = "S6_set2.csv"
LABELS_FILE      = "labels.csv"
K_FEATURES       = 8
K_NEIGHBORS      = 3
TEST_SIZE        = 0.5
RANDOM_SEED      = 42
N_MCCV           = 100   # liczba powtórzeń MCCV
STABILITY_THRESH = 0.5   # próg częstości dla cechy "stabilnej"

# ===========================================================================
# WCZYTANIE DANYCH
# ===========================================================================
def load_dataset(data_file: str, labels_file: str):
    X = pd.read_csv(data_file, header=None).values
    y = pd.read_csv(labels_file, header=None).values.ravel()
    print(f"Załadowano: {data_file}")
    print(f"  Próbki : {X.shape[0]}  (klas 0: {(y==0).sum()}, klas 1: {(y==1).sum()})")
    print(f"  Cechy  : {X.shape[1]}")
    return X, y

# ===========================================================================
# HELPERS
# ===========================================================================
def get_clf():
    return KNeighborsClassifier(n_neighbors=K_NEIGHBORS)

def select_on_all(X, y, k=K_FEATURES):
    sel = SelectKBest(f_classif, k=k)
    X_sel = sel.fit_transform(X, y)
    return X_sel, sel.get_support(indices=True)

def select_on_train(X_train, y_train, X_test, k=K_FEATURES):
    sel = SelectKBest(f_classif, k=k)
    X_train_sel = sel.fit_transform(X_train, y_train)
    X_test_sel  = sel.transform(X_test)
    return X_train_sel, X_test_sel, sel.get_support(indices=True)

# ===========================================================================
# DEMONSTRACJA SELEKCJI CECH 
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
# SCENARIUSZE WALIDACJI 
# ===========================================================================
def scenariusz1_resubstytucja(X, y):
    X_sel, idx = select_on_all(X, y)
    clf = get_clf()
    clf.fit(X_sel, y)
    return zero_one_loss(y, clf.predict(X_sel)), idx

def scenariusz2_leaky_holdout(X, y, random_state=RANDOM_SEED):
    X_sel, idx = select_on_all(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=TEST_SIZE, random_state=random_state
    )
    clf = get_clf()
    clf.fit(X_train, y_train)
    return zero_one_loss(y_test, clf.predict(X_test)), idx

def scenariusz3_poprawny_holdout(X, y, random_state=RANDOM_SEED):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state
    )
    X_train_sel, X_test_sel, idx = select_on_train(X_train, y_train, X_test)
    clf = get_clf()
    clf.fit(X_train_sel, y_train)
    return zero_one_loss(y_test, clf.predict(X_test_sel)), idx

# ===========================================================================
# MCCV – punkt c)
# ===========================================================================
def run_mccv(X, y, n_iter=N_MCCV):
    errors_s1     = np.zeros(n_iter)
    errors_s2     = np.zeros(n_iter)
    errors_s3     = np.zeros(n_iter)
    feature_counts = np.zeros(X.shape[1], dtype=int)

    for i in range(n_iter):
        seed = RANDOM_SEED + i

        errors_s1[i], _   = scenariusz1_resubstytucja(X, y)
        errors_s2[i], _   = scenariusz2_leaky_holdout(X, y, random_state=seed)
        errors_s3[i], idx = scenariusz3_poprawny_holdout(X, y, random_state=seed)
        feature_counts[idx] += 1

    freq = feature_counts / n_iter
    return errors_s1, errors_s2, errors_s3, freq

def print_mccv_results(errors_s1, errors_s2, errors_s3, freq):
    print(f"\n  {'Scenariusz':<38} {'Śr. błąd':>9} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"  {'-'*72}")
    for opis, err in [
        ("Scen. 1 – Resubstytucja",    errors_s1),
        ("Scen. 2 – Leaky holdout",    errors_s2),
        ("Scen. 3 – Poprawny holdout", errors_s3),
    ]:
        print(f"  {opis:<38} {err.mean():>9.4f} {err.std():>7.4f} "
              f"{err.min():>7.4f} {err.max():>7.4f}")

    stable_idx  = np.where(freq >= STABILITY_THRESH)[0]
    stable_freq = freq[stable_idx]
    order       = np.argsort(stable_freq)[::-1]
    stable_idx  = stable_idx[order]
    stable_freq = stable_freq[order]

    print(f"\n  Cechy stabilne (wybrane w >{STABILITY_THRESH*100:.0f}% iteracji, Scen. 3):")
    if len(stable_idx) == 0:
        print(f"    Brak – żadna cecha nie przekroczyła progu")
    else:
        print(f"    Liczba : {len(stable_idx)}")
        print(f"    Indeksy: {stable_idx.tolist()}")
        print(f"    Częst. : {[round(f, 2) for f in stable_freq]}")

    return stable_idx, stable_freq

# ===========================================================================
# GŁÓWNA FUNKCJA
# ===========================================================================
def main():

    # -----------------------------------------------------------------------
    # PUNKT a) – Wybór metod
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("  Punkt a) – Wybór metod selekcji cech i klasyfikacji")
    print("=" * 60)
    print(f"\n  Selekcja cech : ANOVA F-score (SelectKBest, k={K_FEATURES})")
    print(f"  Klasyfikator  : KNN (k={K_NEIGHBORS})\n")

    for label, data_file in [("set1", SET1_FILE), ("set2", SET2_FILE)]:
        print(f"\n{'─'*60}")
        print(f"  Zbiór: {label} ({data_file})")
        print(f"{'─'*60}")
        try:
            X, y = load_dataset(data_file, LABELS_FILE)
        except FileNotFoundError:
            print(f"  [POMINIĘTO] Brak pliku: {data_file}")
            continue
        selector, _ = demo_feature_selection(X, y)
        demo_classifier(selector.transform(X), y)

    print("\n" + "=" * 60)
    print("  Wybór metod zakończony.")

    # -----------------------------------------------------------------------
    # B - Trzy scenariusze, jedno losowanie
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Punkt b) – Trzy scenariusze walidacji (jedno losowanie)")
    print(f"  Podział: {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test, seed={RANDOM_SEED}")
    print("=" * 60)

    wyniki_b = {}
    for nazwa, data_file in [("set1", SET1_FILE), ("set2", SET2_FILE)]:
        print(f"\n{'─'*60}")
        print(f"  Zbiór: {nazwa} ({data_file})")
        print(f"{'─'*60}")
        try:
            X, y = load_dataset(data_file, LABELS_FILE)
        except FileNotFoundError:
            print(f"  [POMINIĘTO] Brak pliku: {data_file}")
            continue

        b1, idx1 = scenariusz1_resubstytucja(X, y)
        b2, idx2 = scenariusz2_leaky_holdout(X, y)
        b3, idx3 = scenariusz3_poprawny_holdout(X, y)
        wyniki_b[nazwa] = {"s1": b1, "s2": b2, "s3": b3}

        print(f"\n  {'Scenariusz':<38} {'Błąd':>6}")
        print(f"  {'-'*46}")
        print(f"  {'Scen. 1 – Resubstytucja (Rys. 2)':<38} {b1:>6.4f}")
        print(f"  {'Scen. 2 – Leaky holdout (Rys. 3)':<38} {b2:>6.4f}")
        print(f"  {'Scen. 3 – Poprawny holdout (Rys. 4)':<38} {b3:>6.4f}")
        print(f"\n  Wybrane cechy (indeksy):")
        print(f"    Scen. 1: {idx1.tolist()}")
        print(f"    Scen. 2: {idx2.tolist()}")
        print(f"    Scen. 3: {idx3.tolist()}")

    if len(wyniki_b) == 2:
        print(f"\n{'='*60}")
        print("  TABELA ZBIORCZA – punkt b)")
        print(f"{'='*60}")
        print(f"  {'Scenariusz walidacji':<38} {'set1':>6} {'set2':>6}")
        print(f"  {'-'*52}")
        for key, opis in [("s1", "Scen. 1 – Resubstytucja"),
                           ("s2", "Scen. 2 – Leaky holdout"),
                           ("s3", "Scen. 3 – Poprawny holdout")]:
            print(f"  {opis:<38} {wyniki_b['set1'][key]:>6.4f} {wyniki_b['set2'][key]:>6.4f}")

    # -----------------------------------------------------------------------
    #  C – MCCV
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Punkt c) – MCCV ({N_MCCV} iteracji)")
    print(f"  Każda iteracja: losowy podział → selekcja cech → KNN → błąd")
    print("=" * 60)

    wyniki_mccv = {}
    for nazwa, data_file in [("set1", SET1_FILE), ("set2", SET2_FILE)]:
        print(f"\n{'─'*60}")
        print(f"  Zbiór: {nazwa} ({data_file})")
        print(f"{'─'*60}")
        try:
            X, y = load_dataset(data_file, LABELS_FILE)
        except FileNotFoundError:
            print(f"  [POMINIĘTO] Brak pliku: {data_file}")
            continue

        print(f"  Uruchamianie MCCV ({N_MCCV} iteracji)...", end=" ", flush=True)
        e1, e2, e3, freq = run_mccv(X, y, n_iter=N_MCCV)
        print("gotowe.")

        wyniki_mccv[nazwa] = (e1, e2, e3, freq)
        print_mccv_results(e1, e2, e3, freq)

    # Tabela zbiorcza MCCV
    if len(wyniki_mccv) == 2:
        print(f"\n{'='*60}")
        print(f"  TABELA ZBIORCZA – punkt c) MCCV (średni błąd ± std)")
        print(f"{'='*60}")
        print(f"  {'Scenariusz walidacji':<32} {'set1':>16} {'set2':>16}")
        print(f"  {'-'*66}")
        for opis, idx in [("Scen. 1 – Resubstytucja",    0),
                           ("Scen. 2 – Leaky holdout",    1),
                           ("Scen. 3 – Poprawny holdout", 2)]:
            e_s1 = wyniki_mccv["set1"][idx]
            e_s2 = wyniki_mccv["set2"][idx]
            print(f"  {opis:<32} {e_s1.mean():.3f} ± {e_s1.std():.3f}  "
                  f"{e_s2.mean():.3f} ± {e_s2.std():.3f}")

        # Identyfikacja zbioru z cechami dyskryminatywnymi
        e3_s1 = wyniki_mccv["set1"][2].mean()
        e3_s2 = wyniki_mccv["set2"][2].mean()
        real  = "set1" if e3_s1 < e3_s2 else "set2"
        rand  = "set2" if real == "set1" else "set1"

        print(f"\n{'='*60}")
        print(f"  WNIOSKI (punkty e, f)")
        print(f"{'='*60}")
        print(f"  Zbiór z różnicującymi cechami (klasy 0≠1) : {real}")
        print(f"  Zbiór losowy (klasy nierozróżnialne)       : {rand}")
        print(f"\n  Wyjaśnienie:")
        print(f"    {real}: Scen. 3 błąd = {wyniki_mccv[real][2].mean():.3f}"
              f"  → klasyfikator znacząco lepszy niż losowy")
        print(f"    {rand}: Scen. 3 błąd = {wyniki_mccv[rand][2].mean():.3f}"
              f"  → bliskie 0.5, czyli losowe zgadywanie")
        print(f"\n  Scen. 1 i 2 fałszują wynik dla OBU zbiorów przez wyciek.")

        # Cechy dyskryminatywne
        _, _, _, freq_real = wyniki_mccv[real]
        stable = np.where(freq_real >= STABILITY_THRESH)[0]
        order  = np.argsort(freq_real[stable])[::-1]
        stable = stable[order]
        print(f"\n  Cechy dyskryminatywne ({real}, próg {STABILITY_THRESH*100:.0f}%):")
        print(f"    Indeksy : {stable.tolist()}")
        print(f"    Częst.  : {[round(freq_real[i], 2) for i in stable]}")

if __name__ == "__main__":
    main()