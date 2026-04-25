"""Simplified student-friendly classifier committees demo.

This single-file script is intentionally easy to read while preserving
the exact outputs (plot filenames and printed summary) from the previous
implementation. It implements:
- Synthetic non-linear two-class data (make_moons)
- Light tuning (validation) for KNN/DecisionTree/SVM/Logistic
- Committees for N in [3,5,7]: bagging, diverse, param-variation (KNN)
- Majority voting and weighted voting (weights = validation accuracy)
- Evaluation: accuracy, confusion matrices (normalized chart) and ROC AUC
- Attempts to load external MAT files `data7.mat` or `dane7.mat` (MATLAB struct with `uczacy`/`testowy`)

Plots are saved to `plots/` (no interactive windows).

Run:
    python simplified_committees.py

Dependencies: numpy, scipy, scikit-learn, matplotlib
"""

import os
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
RANDOM_STATE = 42


# ------------------ Data utilities ------------------

def generate_synthetic(seed: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=1200, noise=0.30, random_state=seed)
    return X, y


def load_mat_if_exists(paths=("data7.mat", "dane7.mat")) -> Tuple[np.ndarray, np.ndarray]:
    for p in paths:
        if os.path.exists(p):
            m = sio.loadmat(p)
            if "uczacy" in m and "testowy" in m:
                tr = m["uczacy"][0, 0]
                te = m["testowy"][0, 0]
                Xtr = np.asarray(tr["X"])
                Xte = np.asarray(te["X"])
                # common MAT stores features x samples -> transpose when needed
                if Xtr.ndim == 2 and Xtr.shape[0] > Xtr.shape[1]:
                    Xtr = Xtr.T
                if Xte.ndim == 2 and Xte.shape[0] > Xte.shape[1]:
                    Xte = Xte.T
                ytr = np.asarray(tr["D"]).ravel().astype(int)
                yte = np.asarray(te["D"]).ravel().astype(int)
                X = np.vstack([Xtr, Xte])
                y = np.concatenate([ytr, yte])
                return X, y
    return None, None


# ------------------ Candidate pools & tuning ------------------

def get_candidate_pools() -> Dict[str, List[Any]]:
    knn = [Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=k))]) for k in (3, 5, 7)]
    tree = [DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE) for d in (3, 5, None)]
    svm = [Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=c, kernel="rbf", probability=True))]) for c in (0.5, 1.0)]
    logreg = [Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])]
    return {"KNN": knn, "DecisionTree": tree, "SVM": svm, "Logistic": logreg}


def tune_select_best(candidates: List[Any], X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, float]:
    best_m = None
    best_acc = -math.inf
    for c in candidates:
        m = clone(c)
        m.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, m.predict(X_val))
        if acc > best_acc:
            best_acc = acc
            best_m = m
    return best_m, best_acc


# ------------------ Ensemble builders ------------------

def build_bagging(base_model: Any, N: int, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int = RANDOM_STATE) -> Tuple[List[Any], np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X_tr.shape[0]
    members = []
    weights = []
    for _ in range(N):
        idx = rng.integers(0, n, n)
        m = clone(base_model)
        m.fit(X_tr[idx], y_tr[idx])
        members.append(m)
        weights.append(accuracy_score(y_val, m.predict(X_val)))
    return members, np.array(weights)


def build_diverse(best_models: List[Tuple[str, Any, float]], N: int, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[List[Any], np.ndarray]:
    pool = []
    for name, model, _ in best_models:
        m = clone(model)
        m.fit(X_tr, y_tr)
        val_acc = accuracy_score(y_val, m.predict(X_val))
        pool.append((name, m, val_acc))
    pool.sort(key=lambda t: t[2], reverse=True)
    sel = pool[:N]
    members = [m for _, m, _ in sel]
    weights = np.array([acc for _, _, acc in sel])
    return members, weights


def build_param_knn(N: int, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[List[Any], np.ndarray]:
    ks = [1, 3, 5, 7, 9, 11, 15][:N]
    members = []
    weights = []
    for k in ks:
        m = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=k))])
        m.fit(X_tr, y_tr)
        members.append(m)
        weights.append(accuracy_score(y_val, m.predict(X_val)))
    return members, np.array(weights)


# ------------------ Voting ------------------

def majority_vote(members: List[Any], X: np.ndarray) -> np.ndarray:
    preds = np.vstack([m.predict(X) for m in members])
    maj = np.apply_along_axis(lambda col: np.bincount(col.astype(int)).argmax(), axis=0, arr=preds)
    return maj


def weighted_vote(members: List[Any], weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    # try to use predict_proba when possible
    proba_list = []
    for m in members:
        if hasattr(m, 'predict_proba'):
            proba_list.append(m.predict_proba(X)[:, 1])
        else:
            proba_list = []
            break
    if proba_list:
        w = weights.astype(float)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()
        agg = np.average(np.vstack(proba_list), axis=0, weights=w)
        return (agg >= 0.5).astype(int)
    # discrete weighted votes fallback
    preds = np.vstack([m.predict(X) for m in members])
    n_samples = X.shape[0]
    n_classes = int(preds.max()) + 1
    scores = np.zeros((n_samples, n_classes))
    for w, p in zip(weights, preds):
        for i, cls in enumerate(p):
            scores[i, int(cls)] += w
    return np.argmax(scores, axis=1)


# ------------------ Evaluation & plotting ------------------

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {"accuracy": accuracy_score(y_true, y_pred), "confusion_matrix": confusion_matrix(y_true, y_pred)}


def save_confusion_chart(cm: np.ndarray, label: str, normalize: bool = True) -> str:
    lab = label.replace(' ', '_')
    fig, ax = plt.subplots(figsize=(4, 3))
    cmf = cm.astype(float)
    if normalize:
        rows = cmf.sum(axis=1, keepdims=True)
        pct = np.zeros_like(cmf, dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            np.divide(cmf, rows, out=pct, where=(rows != 0))
        annot = np.array([[f"{int(cmf[i,j])}\n{pct[i,j]*100:.1f}%" for j in range(cmf.shape[1])] for i in range(cmf.shape[0])])
    else:
        annot = np.array([[f"{int(cmf[i,j])}" for j in range(cmf.shape[1])] for i in range(cmf.shape[0])])
    im = ax.imshow(cmf, cmap='Blues')
    for i in range(cmf.shape[0]):
        for j in range(cmf.shape[1]):
            ax.text(j, i, annot[i, j], ha='center', va='center', color='white' if cmf[i, j] > cmf.max()/2 else 'black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(label)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = os.path.join(PLOTS_DIR, f'cmchart_{lab}.png')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def save_roc_plot(y_true: np.ndarray, scores: np.ndarray, label: str) -> str:
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
    except Exception:
        return ''
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(label)
    ax.legend(loc='lower right')
    out = os.path.join(PLOTS_DIR, f'roc_{label.replace(" ", "_")}.png')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_dataset_2d(X: np.ndarray, y: np.ndarray, title: str, fname: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20, edgecolor='k')
    ax.set_title(title)
    out = os.path.join(PLOTS_DIR, fname)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ------------------ Committee evaluation helper ------------------

def _aggregate_probs(members: List[Any], X: np.ndarray) -> List[np.ndarray]:
    probs = []
    for m in members:
        if hasattr(m, 'predict_proba'):
            probs.append(m.predict_proba(X)[:, 1])
    return probs


def evaluate_committee(members: List[Any], weights: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, dataset: str, kind: str, N: int) -> Dict[str, Any]:
    pred_maj = majority_vote(members, X_test)
    maj_res = evaluate_predictions(y_test, pred_maj)
    cm1 = save_confusion_chart(maj_res['confusion_matrix'], f"{dataset}_{kind}_{N}_majority")

    pred_w = weighted_vote(members, weights, X_test)
    w_res = evaluate_predictions(y_test, pred_w)
    cm2 = save_confusion_chart(w_res['confusion_matrix'], f"{dataset}_{kind}_{N}_weighted")

    probs = _aggregate_probs(members, X_test)
    if probs:
        agg = np.average(np.vstack(probs), axis=0, weights=weights if weights.sum() > 0 else None)
        roc_path = save_roc_plot(y_test, agg, f"{dataset}_{kind}_{N}")
    else:
        roc_path = ''

    return {"type": kind, "N": N, "maj": maj_res, "weighted": w_res, "cm_path": cm1, "cmw_path": cm2, "roc_path": roc_path}


# ------------------ Full pipeline ------------------

def run_full_pipeline(X: np.ndarray, y: np.ndarray, dataset_name: str = 'synthetic') -> Dict[str, Any]:
    # splits
    X_trv, X_test, y_trv, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(X_trv, y_trv, test_size=0.25, stratify=y_trv, random_state=RANDOM_STATE)

    # plot dataset
    if X.shape[1] == 2:
        plot_dataset_2d(X, y, f"Dataset: {dataset_name}", f"{dataset_name}_dataset.png")
    else:
        X2 = PCA(n_components=2).fit_transform(X)
        plot_dataset_2d(X2, y, f"PCA scatter: {dataset_name}", f"{dataset_name}_pca_dataset.png")

    # tune
    pools = get_candidate_pools()
    best_models = []
    for fam, cands in pools.items():
        m, acc = tune_select_best(cands, X_tr, y_tr, X_val, y_val)
        best_models.append((fam, m, acc))

    # evaluate singles
    singles = []
    for name, model, val_acc in best_models:
        test_acc = accuracy_score(y_test, model.predict(X_test))
        cm = confusion_matrix(y_test, model.predict(X_test))
        cm_path = save_confusion_chart(cm, f"{dataset_name}_single_{name}", normalize=True)
        roc_path = ''
        try:
            if hasattr(model, 'predict_proba'):
                scores = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test)
            else:
                scores = model.predict(X_test)
            roc_path = save_roc_plot(y_test, scores, f"{dataset_name}_single_{name}")
        except Exception:
            pass
        singles.append({"name": name, "val_acc": val_acc, "test_acc": test_acc, "cm": cm, "cm_path": cm_path, "roc_path": roc_path})

    # committees
    N_values = [3, 5, 7]
    committees = []

    # bagging (DecisionTree base)
    dt_model = next((m for fam, m, _ in best_models if fam == 'DecisionTree'), None)
    if dt_model is None:
        dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
        dt_model.fit(X_tr, y_tr)
    for N in N_values:
        members, weights = build_bagging(dt_model, N, X_tr, y_tr, X_val, y_val)
        res = evaluate_committee(members, weights, X_test, y_test, dataset_name, 'bagging', N)
        committees.append(res)

    # diverse
    sorted_best = sorted(best_models, key=lambda t: t[2], reverse=True)
    for N in N_values:
        sel = sorted_best[:N]
        members = [clone(m).fit(X_tr, y_tr) for _, m, _ in sel]
        weights = np.array([accuracy_score(y_val, m.predict(X_val)) for _, m, _ in sel])
        res = evaluate_committee(members, weights, X_test, y_test, dataset_name, 'diverse', N)
        committees.append(res)

    # param variation (KNN)
    for N in N_values:
        members, weights = build_param_knn(N, X_tr, y_tr, X_val, y_val)
        res = evaluate_committee(members, weights, X_test, y_test, dataset_name, 'paramknn', N)
        committees.append(res)

    # print summaries
    print("\n=== Single classifiers ===")
    for s in singles:
        print(f"{dataset_name} - {s['name']:12s} val_acc={s['val_acc']:.3f} test_acc={s['test_acc']:.3f} cm={s['cm_path']} roc={s['roc_path']}")

    print("\n=== Committees summary ===")
    for c in committees:
        print(f"{dataset_name} - {c['type']:8s} N={c['N']:d} majority_acc={c['maj']['accuracy']:.3f} weighted_acc={c['weighted']['accuracy']:.3f} cm={c['cm_path']} cmw={c['cmw_path']} roc={c['roc_path']}")

    # analysis by N
    analysis = {}
    for kind in ['bagging', 'diverse', 'paramknn']:
        rows = [c for c in committees if c['type'] == kind]
        rows.sort(key=lambda r: r['N'])
        analysis[kind] = [(r['N'], r['maj']['accuracy'], r['weighted']['accuracy']) for r in rows]

    best_single = max(singles, key=lambda x: x['test_acc'])
    best_committee = max(committees, key=lambda c: max(c['maj']['accuracy'], c['weighted']['accuracy']))

    return {
        'dataset': dataset_name,
        'singles': singles,
        'committees': committees,
        'analysis': analysis,
        'best_single': best_single,
        'best_committee': best_committee,
    }


def main():
    print("Running pipeline on synthetic data...")
    Xs, ys = generate_synthetic()
    summary_syn = run_full_pipeline(Xs, ys, dataset_name='synthetic')

    X_ext, y_ext = load_mat_if_exists()
    summary_ext = None
    if X_ext is not None:
        print("\nExternal MAT detected — running pipeline on external data...")
        summary_ext = run_full_pipeline(X_ext, y_ext, dataset_name='external')
    else:
        print("\nNo external MAT found (data7.mat / dane7.mat). Skipping external run.")

    print("\n=== Final Short Summary ===")
    print(f"Best single on synthetic: {summary_syn['best_single']['name']} (test_acc={summary_syn['best_single']['test_acc']:.3f})")
    bc = summary_syn['best_committee']
    print(f"Best committee on synthetic: type={bc['type']}, N={bc['N']}, acc=max({bc['maj']['accuracy']:.3f},{bc['weighted']['accuracy']:.3f}))")

    if summary_ext:
        print(f"Best single on external: {summary_ext['best_single']['name']} (test_acc={summary_ext['best_single']['test_acc']:.3f})")
        bc2 = summary_ext['best_committee']
        print(f"Best committee on external: type={bc2['type']}, N={bc2['N']}, acc=max({bc2['maj']['accuracy']:.3f},{bc2['weighted']['accuracy']:.3f}))")

    print("\nAnalysis: Increasing N sometimes helps (depends on diversity and base strength). Weighted voting can improve when member accuracies vary; majority may fail when many weak but correlated members outvote a smaller set of strong members. See saved CM and ROC plots in the plots/ folder for details.")


if __name__ == '__main__':
    main()
