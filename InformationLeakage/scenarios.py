"""
scenarios.py – trzy scenariusze walidacji + MCCV + funkcje wydruku
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss

from data import (
    get_clf, select_on_all, select_on_train,
    TEST_SIZE, RANDOM_SEED, N_MCCV, STABILITY_THRESH, K_FEATURES
)

# ===========================================================================
# SCENARIUSZ 1 – RESUBSTYTUCJA 
# Dane → Selekcja Cech → Trening i Testowanie na tych samych danych
# Wyciek: MAKSYMALNY
# ===========================================================================
def scenariusz1_resubstytucja(X, y):
    X_sel, idx = select_on_all(X, y)
    clf = get_clf()
    clf.fit(X_sel, y)
    return zero_one_loss(y, clf.predict(X_sel)), idx

# ===========================================================================
# SCENARIUSZ 2 – HOLDOUT Z WYCIEKIEM 
# Dane → Selekcja Cech → Podział → Trening / Testowanie
# Wyciek: selekcja cech uwzględnia przyszły zbiór testowy
# ===========================================================================
def scenariusz2_leaky_holdout(X, y, random_state=RANDOM_SEED):
    X_sel, idx = select_on_all(X, y)          # selekcja na CAŁOŚCI – WYCIEK!
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=TEST_SIZE, random_state=random_state
    )
    clf = get_clf()
    clf.fit(X_train, y_train)
    return zero_one_loss(y_test, clf.predict(X_test)), idx

# ===========================================================================
# SCENARIUSZ 3 – POPRAWNY HOLDOUT 
# Dane → Podział → Training set → Selekcja Cech → Trenowanie
#                → Test set     → (tylko transform) → Testowanie
# Brak wycieku: test odizolowany przed jakimkolwiek przetwarzaniem
# ===========================================================================
def scenariusz3_poprawny_holdout(X, y, random_state=RANDOM_SEED):
    X_train, X_test, y_train, y_test = train_test_split(   # podział JAKO PIERWSZY
        X, y, test_size=TEST_SIZE, random_state=random_state
    )
    X_train_sel, X_test_sel, idx = select_on_train(X_train, y_train, X_test)
    clf = get_clf()
    clf.fit(X_train_sel, y_train)
    return zero_one_loss(y_test, clf.predict(X_test_sel)), idx

# ===========================================================================
# MCCV – Monte Carlo Cross-Validation
# ===========================================================================
def run_mccv(X, y, n_iter=N_MCCV):
    errors_s1      = np.zeros(n_iter)
    errors_s2      = np.zeros(n_iter)
    errors_s3      = np.zeros(n_iter)
    feature_counts = np.zeros(X.shape[1], dtype=int)

    for i in range(n_iter):
        seed = RANDOM_SEED + i
        errors_s1[i], _   = scenariusz1_resubstytucja(X, y)
        errors_s2[i], _   = scenariusz2_leaky_holdout(X, y, random_state=seed)
        errors_s3[i], idx = scenariusz3_poprawny_holdout(X, y, random_state=seed)
        feature_counts[idx] += 1

    freq = feature_counts / n_iter
    return errors_s1, errors_s2, errors_s3, freq

# ===========================================================================
# WYDRUKI
# ===========================================================================
def print_scenario_results(b1, idx1, b2, idx2, b3, idx3):
    print(f"\n  {'Scenariusz':<38} {'Błąd':>6}")
    print(f"  {'-'*46}")
    print(f"  {'Scen. 1 – Resubstytucja (Rys. 2)':<38} {b1:>6.4f}")
    print(f"  {'Scen. 2 – Leaky holdout (Rys. 3)':<38} {b2:>6.4f}")
    print(f"  {'Scen. 3 – Poprawny holdout (Rys. 4)':<38} {b3:>6.4f}")
    print(f"\n  Wybrane cechy (indeksy):")
    print(f"    Scen. 1: {idx1.tolist()}")
    print(f"    Scen. 2: {idx2.tolist()}")
    print(f"    Scen. 3: {idx3.tolist()}")

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
        print(f"    Częst. : {[round(float(f), 2) for f in stable_freq]}")

    return stable_idx, stable_freq