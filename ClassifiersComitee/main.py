import numpy as np
from sklearn.model_selection import train_test_split

from data import generuj_dane, wczytaj_mat
from classifiers import (
    ocen_kandydatow, komitet_bagging, komitet_roznorodny, komitet_knn_param, ocen_komitet,)
from plots import (
    wykres_danych, wykres_kandydatow, wykres_komitetu,)

RNG = 42

# Glowny watek - dla danych syntetycznych i dane7.mat
def _rdzen(Xtr, ytr, Xval, yval, Xtest, ytest, nazwa):

    print(f"\n{'='*55}")
    print(f"  DATASET: {nazwa}")
    print(f"  Uczący : {Xtr.shape[0]} próbek | Testowy: {Xtest.shape[0]} próbek | Cech: {Xtr.shape[1]}")
    print(f"{'='*55}")
 
    # Wykres 1 – scatter danych 
    wykres_danych(np.vstack([Xtr, Xtest]), np.concatenate([ytr, ytest]), nazwa)
 
    # Zadanie 2: dobór parametrów 
    print("\n--- Dobór parametrów ---")
    posortowani = ocen_kandydatow(Xtr, ytr, Xval, yval)
    for nk, _, acc in posortowani:
        print(f"  {nk:12s}  acc doboru={acc:.3f}")
 
    najlepsza_nazwa = posortowani[0][0]
    print(f"  >> Najlepszy kandydat: {najlepsza_nazwa} (używany w 3b diverse)")
 
    wykres_kandydatow(posortowani, Xtest, ytest, nazwa)
 
    # Zadanie 3: komitety dla N = 3, 5, 7 
    print("\n--- Komitety demokratyczne ---")
    wyniki = []
 
    for N in [3, 5, 7]:
        print(f"\n  N = {N}")
 
        # 3a – ten sam klasyfikator (DecisionTree), różne dane uczące 
        czl, wag = komitet_bagging(N, Xtr, ytr, Xval, yval)
        maj, waz, ada = ocen_komitet(czl, wag, Xval, yval, Xtest, ytest)
        wyniki.append(("bagging", N, maj, waz, ada))
        print(f"    3a bagging     większość={maj:.3f}  ważone={waz:.3f}  adaptacyjne={ada:.3f}")
 
        # 3b – różne klasyfikatory, te same dane
        czl, wag = komitet_roznorodny(posortowani, N, Xtr, ytr, Xval, yval)
        maj, waz, ada = ocen_komitet(czl, wag, Xval, yval, Xtest, ytest)
        wyniki.append(("diverse", N, maj, waz, ada))
        print(f"    3b diverse     większość={maj:.3f}  ważone={waz:.3f}  adaptacyjne={ada:.3f}")
 
        # 3c – KNN z różnymi k, te same dane
        czl, wag = komitet_knn_param(N, Xtr, ytr, Xval, yval)
        maj, waz, ada = ocen_komitet(czl, wag, Xval, yval, Xtest, ytest)
        wyniki.append(("knn_param", N, maj, waz, ada))
        print(f"    3c knn_param   większość={maj:.3f}  ważone={waz:.3f}  adaptacyjne={ada:.3f}")
 
    # Wykresy 3a / 3b / 3c 
    wykres_komitetu(wyniki, "bagging",   "3a) Ten sam klasyfikator, różne dane uczące", nazwa, "3a")
    wykres_komitetu(wyniki, "diverse",   "3b) Różne klasyfikatory, te same dane",       nazwa, "3b")
    wykres_komitetu(wyniki, "knn_param", "3c) KNN z różnymi k, te same dane",           nazwa, "3c")
 

# Pipeline 1
def pipeline_syntetyczne(X, y, nazwa="syntetyczne"):
    Xtrv, Xtest, ytrv, ytest = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG)
    Xtr,  Xval,  ytr,  yval  = train_test_split(Xtrv, ytrv, test_size=0.20, stratify=ytrv, random_state=RNG)
    _rdzen(Xtr, ytr, Xval, yval, Xtest, ytest, nazwa)
 
# Pipeline 2 - dane7.mat
def pipeline_zewnetrzne(Xtr, ytr, Xte, yte, nazwa="zewnetrzne"):
    _rdzen(Xtr, ytr, Xtr, ytr, Xte, yte, nazwa)
 

def main():
    # 1 – dane syntetyczne
    print("Generowanie danych syntetycznych (make_moons)...")
    X, y = generuj_dane()
    pipeline_syntetyczne(X, y, nazwa="syntetyczne")

    # 2 – dane rzeczywiste z pliku MAT
    print("\nSzukam pliku data7.mat / dane7.mat...")
    Xtr, ytr, Xte, yte = wczytaj_mat()
    if Xtr is not None:
        pipeline_zewnetrzne(Xtr, ytr, Xte, yte, nazwa="zewnetrzne")
    else:
        print("  Brak pliku MAT – pomijam dane zewnętrzne.")


if __name__ == "__main__":
    main()