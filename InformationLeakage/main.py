"""
zadanie1 – Ćwiczenie 5: Przeciek informacji i jego unikanie
Uruchomienie: python main.py
"""

import numpy as np
from data import (
    load_dataset, load_dataset2, demo_feature_selection, demo_classifier, 
    SET1_FILE, SET2_FILE, LABELS_FILE, EXT_SET_FILE,
    K_FEATURES, K_NEIGHBORS, TEST_SIZE, RANDOM_SEED, N_MCCV, STABILITY_THRESH
)
from scenarios import (
    scenariusz1_resubstytucja,
    scenariusz2_leaky_holdout,
    scenariusz3_poprawny_holdout,
    run_mccv,
    print_scenario_results,
    print_mccv_results,
)
from plots import (
    plot_feature_importance,
    plot_mccv_boxplot,
    plot_feature_stability,
    plot_mean_errors_comparison,
    plot_mccv_boxplot_single
)

DATASETS = [("set1", SET1_FILE), ("set2", SET2_FILE)]

def zadanie1():

    # -----------------------------------------------------------------------
    # Wybór metod selekcji cech i klasyfikacji
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(" Wybór metod selekcji cech i klasyfikacji")
    print("=" * 60)
    print(f"\n  Selekcja cech : ANOVA F-score (SelectKBest, k={K_FEATURES})")
    print(f"  Klasyfikator  : KNN (k={K_NEIGHBORS})\n")

    for label, data_file in DATASETS:
        print(f"\n{'─'*60}")
        print(f"  Zbiór: {label} ({data_file})")
        print(f"{'─'*60}")
        try:
            X, y = load_dataset(data_file)
        except FileNotFoundError:
            print(f"  [POMINIĘTO] Brak pliku: {data_file}")
            continue
        selector, _ = demo_feature_selection(X, y)
        demo_classifier(selector.transform(X), y)

        # Wykres F-score z zaznaczonym progiem k=5
        plot_feature_importance(X, y, label, k_selected=K_FEATURES)

    print("\n" + "=" * 60)
    print("  Wybór metod zakończony.")

    # -----------------------------------------------------------------------
    # Trzy scenariusze walidacji (jedno losowanie)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Trzy scenariusze walidacji (jedno losowanie)")
    print(f"  Podział: {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test, seed={RANDOM_SEED}")
    print("=" * 60)

    wyniki_b = {}
    for nazwa, data_file in DATASETS:
        print(f"\n{'─'*60}")
        print(f"  Zbiór: {nazwa} ({data_file})")
        print(f"{'─'*60}")
        try:
            X, y = load_dataset(data_file)
        except FileNotFoundError:
            print(f"  [POMINIĘTO] Brak pliku: {data_file}")
            continue

        b1, idx1 = scenariusz1_resubstytucja(X, y)
        b2, idx2 = scenariusz2_leaky_holdout(X, y)
        b3, idx3 = scenariusz3_poprawny_holdout(X, y)
        wyniki_b[nazwa] = {"s1": b1, "s2": b2, "s3": b3}
        print_scenario_results(b1, idx1, b2, idx2, b3, idx3)

    if len(wyniki_b) == 2:
        print(f"\n{'='*60}")
        print("  TABELA ZBIORCZA)")
        print(f"{'='*60}")
        print(f"  {'Scenariusz walidacji':<38} {'set1':>6} {'set2':>6}")
        print(f"  {'-'*52}")
        for key, opis in [("s1", "Scen. 1 – Resubstytucja"),
                           ("s2", "Scen. 2 – Leaky holdout"),
                           ("s3", "Scen. 3 – Poprawny holdout")]:
            print(f"  {opis:<38} {wyniki_b['set1'][key]:>6.4f} {wyniki_b['set2'][key]:>6.4f}")

    # -----------------------------------------------------------------------
    # MCCV
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  MCCV ({N_MCCV} iteracji)")
    print(f"  Każda iteracja: losowy podział → selekcja cech → KNN → błąd")
    print("=" * 60)

    wyniki_mccv = {}
    for nazwa, data_file in DATASETS:
        print(f"\n{'─'*60}")
        print(f"  Zbiór: {nazwa} ({data_file})")
        print(f"{'─'*60}")
        try:
            X, y = load_dataset(data_file)
        except FileNotFoundError:
            print(f"  [POMINIĘTO] Brak pliku: {data_file}")
            continue

        print(f"  Uruchamianie MCCV ({N_MCCV} iteracji)...", end=" ", flush=True)
        e1, e2, e3, freq = run_mccv(X, y, n_iter=N_MCCV)
        print("gotowe.")

        wyniki_mccv[nazwa] = (e1, e2, e3, freq)
        print_mccv_results(e1, e2, e3, freq)

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

        # Identyfikacja zbioru z sygnałem
        e3_s1 = wyniki_mccv["set1"][2].mean()
        e3_s2 = wyniki_mccv["set2"][2].mean()
        real  = "set1" if e3_s1 < e3_s2 else "set2"
        rand  = "set2" if real == "set1" else "set1"

        # -----------------------------------------------------------------------
        # e,f g
        # -----------------------------------------------------------------------
        print(f"\n{'='*60}")
 
        print(f"{'='*60}")
        print(f"  Zbiór z różnicującymi cechami (klasy 0≠1) : {real}")
        print(f"  Zbiór losowy (klasy nierozróżnialne)       : {rand}")
        print(f"\n  Wyjaśnienie:")
        print(f"    {real}: Scen. 3 błąd = {wyniki_mccv[real][2].mean():.3f}"
              f"  → klasyfikator znacząco lepszy niż losowy")
        print(f"    {rand}: Scen. 3 błąd = {wyniki_mccv[rand][2].mean():.3f}"
              f"  → bliskie 0.5, czyli losowe zgadywanie")

        _, _, _, freq_real = wyniki_mccv[real]
        stable = np.where(freq_real >= STABILITY_THRESH)[0]
        order  = np.argsort(freq_real[stable])[::-1]
        stable = stable[order]
        print(f"\n  Cechy dyskryminatywne {real}, próg {STABILITY_THRESH*100:.0f}%):")
        print(f"    Indeksy : {stable.tolist()}")
        print(f"    Częst.  : {[round(float(freq_real[i]), 2) for i in stable]}")

        # -----------------------------------------------------------------------
        # WYKRESY
        # -----------------------------------------------------------------------
        print(f"\n  Generowanie wykresów...")

        # Boxplot rozkładu błędów MCCV – punkty b, c, e
        plot_mccv_boxplot(wyniki_mccv)

        # Grupowany barplot set1 vs set2 – punkty e, f
        plot_mean_errors_comparison(wyniki_mccv)

        # Stabilność cech – punkt g
        plot_feature_stability(wyniki_mccv, real_set=real)

        print(f"\n  Zapisano wykresy:")
        print(f"    plot_feature_importance_set1.png  – uzasadnienie k=5  (punkt a)")
        print(f"    plot_feature_importance_set2.png  – brak sygnału       (punkt f)")
        print(f"    plot_mccv_boxplot.png             – rozkład błędów     (punkty b/c/e)")
        print(f"    plot_mean_errors_comparison.png   – set1 vs set2       (punkty e/f)")
        print(f"    plot_feature_stability_{real}.png – cechy stabilne     (punkt g)")

def zadanie2():
    """
    Zadanie 2 – Walidacja na zbiorze zewnętrznym (EXT_SET_FILE / sonardata.csv)
    Przeprowadza te same trzy scenariusze walidacji i MCCV co w zadaniu 1,
    ale wyłącznie dla zbioru EXT_SET_FILE (dane sonarowe, 60 cech, etykiety R/M).
    """
    print("\n" + "=" * 60)
    print(" ZADANIE 2 – Zbiór zewnętrzny: sonardata.csv")
    print("=" * 60)

    try:
        X, y = load_dataset2(EXT_SET_FILE)
    except FileNotFoundError:
        print(f"  [POMINIĘTO] Brak pliku: {EXT_SET_FILE}")
        return

    # -------------------------------------------------------------------
    # 2a) Wybór metod – selekcja cech i klasyfikator
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  2a) Selekcja cech i klasyfikator")
    print("-" * 60)
    selector, _ = demo_feature_selection(X, y)
    demo_classifier(selector.transform(X), y)
    plot_feature_importance(X, y, "SONAR", k_selected=K_FEATURES)

    # -------------------------------------------------------------------
    # 2b) Trzy scenariusze walidacji (jedno losowanie)
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("  2b) Trzy scenariusze walidacji (jedno losowanie)")
    print(f"  Podział: {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test, seed={RANDOM_SEED}")
    print("-" * 60)

    b1, idx1 = scenariusz1_resubstytucja(X, y)
    b2, idx2 = scenariusz2_leaky_holdout(X, y)
    b3, idx3 = scenariusz3_poprawny_holdout(X, y)
    print_scenario_results(b1, idx1, b2, idx2, b3, idx3)

    # -------------------------------------------------------------------
    # 2c) MCCV
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(f"  2c) MCCV ({N_MCCV} iteracji)")
    print("-" * 60)
    e1, e2, e3, freq = run_mccv(X, y, n_iter=N_MCCV)
    print_mccv_results(e1, e2, e3, freq)

    wyniki_mccv_ext = {"SONAR": (e1, e2, e3, freq)}
   
    print(f"\n{'='*60}")
    print(f"  TABELA ZBIORCZA – punkt c) MCCV (średni błąd ± std)")
    print(f"{'='*60}")
    print(f"  {'Scenariusz walidacji':<32} {'set1':>16} {'set2':>16}")
    print(f"  {'-'*66}")
    for opis, idx in [("Scen. 1 – Resubstytucja",    0),
                        ("Scen. 2 – Leaky holdout",    1),
                        ("Scen. 3 – Poprawny holdout", 2)]:
        e_s = wyniki_mccv_ext["SONAR"][idx]
        print(f"  {opis:<32} {e_s.mean():.3f} ± {e_s.std():.3f}")

    # -------------------------------------------------------------------
    # 2d) Identyfikacja cech dyskryminatywnych
    # -------------------------------------------------------------------
    print(f"\n  Wyniki MCCV (Scen. 3):")
    print(f"    Błąd = {e3.mean():.3f} ± {e3.std():.3f}")


    stable = np.where(freq >= STABILITY_THRESH)[0]
    order  = np.argsort(freq[stable])[::-1]
    stable = stable[order]
    print(f"\n  Cechy dyskryminatywne (próg {STABILITY_THRESH*100:.0f}%):")
    if len(stable) == 0:
        print(f"    Brak cech przekraczających próg stabilności.")
    else:
        print(f"    Indeksy : {stable.tolist()}")
        print(f"    Częst.  : {[round(float(freq[i]), 2) for i in stable]}")

    # -------------------------------------------------------------------
    # 2e) Wykresy
    # -------------------------------------------------------------------
    plot_mccv_boxplot_single(e1, e2, e3, "SONAR")
    plot_feature_stability(wyniki_mccv_ext, real_set="SONAR")

    print(f"\n  Zapisano wykresy:")
    print(f"    plot_feature_importance_ext_set.png  – istotność cech  (2a)")
    print(f"    plot_mccv_boxplot.png                – rozkład błędów  (2c)")
    print(f"    plot_feature_stability_ext_set.png   – cechy stabilne  (2d)")


if __name__ == "__main__":
    zadanie1()
    zadanie2()