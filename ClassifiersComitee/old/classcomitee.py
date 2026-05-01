import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Committee:
    """Simple ensemble committee supporting majority and weighted voting."""

    def __init__(self, name, estimators, weights, n_classes, kind, n_members):
        self.name = name
        self.estimators = estimators
        self.weights = np.asarray(weights, dtype=float)
        self.n_classes = n_classes
        self.kind = kind
        self.n_members = n_members

    def _member_predictions(self, X):
        preds = [est.predict(X) for est in self.estimators]
        return np.vstack(preds)

    def predict_majority(self, X):
        member_preds = self._member_predictions(X)
        final_preds = []
        for col in member_preds.T:
            counts = np.bincount(col.astype(int), minlength=self.n_classes)
            final_preds.append(np.argmax(counts))
        return np.array(final_preds)

    def predict_weighted(self, X):
        member_preds = self._member_predictions(X)
        scores = np.zeros((X.shape[0], self.n_classes), dtype=float)
        for w, preds in zip(self.weights, member_preds):
            scores[np.arange(X.shape[0]), preds.astype(int)] += w
        return np.argmax(scores, axis=1)


def generate_data(random_state=42):
    """Generate a slightly overlapping non-linear 2-class dataset."""
    X, y = make_moons(n_samples=1200, noise=0.30, random_state=random_state)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=random_state,
    )
    return X, y, X_train, y_train, X_val, y_val, X_test, y_test


def accuracy_to_weight(acc):
    """Convert validation accuracy to a positive voting weight above chance level."""
    return max(acc - 0.50, 1e-6)


def tune_model(name, candidates, X_train, y_train, X_val, y_val):
    """Select best model by validation accuracy from a candidate list."""
    best_model = None
    best_score = -np.inf

    for model in candidates:
        fitted = clone(model).fit(X_train, y_train)
        val_pred = fitted.predict(X_val)
        score = accuracy_score(y_val, val_pred)
        if score > best_score:
            best_score = score
            best_model = fitted

    return name, best_model, best_score


def get_base_candidates(random_state=42):
    knn_candidates = [
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=k, weights=w)),
        ])
        for k in [3, 5, 7, 9, 11]
        for w in ["uniform", "distance"]
    ]

    tree_candidates = [
        DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        for max_depth in [3, 4, 5, 6, None]
        for min_samples_leaf in [1, 3, 5]
    ]

    svm_candidates = [
        Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    probability=False,
                    random_state=random_state,
                ),
            ),
        ])
        for C in [0.5, 1.0, 2.0, 4.0]
        for gamma in ["scale", 0.5, 1.0]
    ]

    logreg_candidates = [
        Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ])
        for C in [0.1, 0.3, 1.0, 3.0, 10.0]
    ]

    return {
        "KNN": knn_candidates,
        "Decision Tree": tree_candidates,
        "SVM": svm_candidates,
        "Logistic Regression": logreg_candidates,
    }


def evaluate_single_classifiers(best_models, X_test, y_test):
    results = []
    for name, model, val_acc in best_models:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results.append(
            {
                "name": name,
                "val_accuracy": val_acc,
                "test_accuracy": acc,
                "confusion_matrix": cm,
                "model": model,
            }
        )
    return results


def build_bagging_committees(n_values, base_model, X_train, y_train, X_val, y_val, seed=42):
    committees = []
    rng = np.random.default_rng(seed)
    n_samples = len(X_train)

    for n in n_values:
        estimators = []
        weights = []
        for _ in range(n):
            idx = rng.integers(0, n_samples, size=n_samples)
            model = clone(base_model).fit(X_train[idx], y_train[idx])
            estimators.append(model)
            val_acc = accuracy_score(y_val, model.predict(X_val))
            weights.append(accuracy_to_weight(val_acc))

        committees.append(
            Committee(
                name=f"Bagging (same classifier, N={n})",
                estimators=estimators,
                weights=weights,
                n_classes=2,
                kind="Bagging (same classifier)",
                n_members=n,
            )
        )

    return committees


def build_diverse_model_pool(base_best_models, random_state=42):
    """Create a heterogeneous pool of classifiers trained on the same data."""
    best_map = {name: model for name, model, _ in base_best_models}

    # Keep the four required tuned classifiers and add three extra model families.
    pool = [
        ("KNN", best_map["KNN"]),
        ("Decision Tree", best_map["Decision Tree"]),
        ("SVM", best_map["SVM"]),
        ("Logistic Regression", best_map["Logistic Regression"]),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=random_state,
            ),
        ),
        (
            "MLP",
            Pipeline([
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(20, 20),
                        alpha=0.001,
                        max_iter=1500,
                        random_state=random_state,
                    ),
                ),
            ]),
        ),
    ]
    return pool


def build_diverse_committees(n_values, model_pool, X_train, y_train, X_val, y_val):
    trained = []
    for name, model in model_pool:
        fitted = clone(model).fit(X_train, y_train)
        val_acc = accuracy_score(y_val, fitted.predict(X_val))
        trained.append((name, fitted, val_acc))

    trained.sort(key=lambda item: item[2], reverse=True)

    committees = []
    for n in n_values:
        selected = trained[:n]
        estimators = [m for _, m, _ in selected]
        weights = [accuracy_to_weight(acc) for _, _, acc in selected]
        committees.append(
            Committee(
                name=f"Diverse classifiers, same data (N={n})",
                estimators=estimators,
                weights=weights,
                n_classes=2,
                kind="Diverse classifiers, same data",
                n_members=n,
            )
        )
    return committees


def build_knn_param_committees(n_values, X_train, y_train, X_val, y_val):
    committees = []
    k_map = {
        3: [1, 11, 31],
        5: [1, 7, 15, 31, 61],
        7: [1, 5, 11, 21, 41, 61, 81],
    }

    for n in n_values:
        estimators = []
        weights = []
        for k in k_map[n]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=k, weights="distance")),
            ])
            fitted = model.fit(X_train, y_train)
            estimators.append(fitted)
            val_acc = accuracy_score(y_val, fitted.predict(X_val))
            weights.append(accuracy_to_weight(val_acc))

        committees.append(
            Committee(
                name=f"Same classifier, different params (KNN), N={n}",
                estimators=estimators,
                weights=weights,
                n_classes=2,
                kind="Same classifier, different params (KNN)",
                n_members=n,
            )
        )

    return committees


def evaluate_committees(committees, X_test, y_test):
    rows = []
    for committee in committees:
        pred_majority = committee.predict_majority(X_test)
        pred_weighted = committee.predict_weighted(X_test)

        rows.append(
            {
                "name": committee.name,
                "voting": "majority",
                "accuracy": accuracy_score(y_test, pred_majority),
                "confusion_matrix": confusion_matrix(y_test, pred_majority),
                "committee": committee,
                "kind": committee.kind,
                "n_members": committee.n_members,
            }
        )
        rows.append(
            {
                "name": committee.name,
                "voting": "weighted",
                "accuracy": accuracy_score(y_test, pred_weighted),
                "confusion_matrix": confusion_matrix(y_test, pred_weighted),
                "committee": committee,
                "kind": committee.kind,
                "n_members": committee.n_members,
            }
        )
    return rows


def print_results(single_results, committee_results):
    print("\n" + "=" * 80)
    print("TUNED SINGLE CLASSIFIERS")
    print("=" * 80)
    for row in single_results:
        print(
            f"{row['name']:<22} val_acc={row['val_accuracy']:.4f} "
            f"test_acc={row['test_accuracy']:.4f}"
        )
        # plot confusion matrix instead of printing
        fig, ax = plt.subplots(figsize=(4, 3))
        plot_confusion_matrix_ax(row["confusion_matrix"], ax, title=f"{row['name']} (single)")

    print("\n" + "=" * 80)
    print("CLASSIFIER COMMITTEES")
    print("=" * 80)
    for row in sorted(committee_results, key=lambda x: x["accuracy"], reverse=True):
        print(f"{row['name']:<45} {row['voting']:<8} acc={row['accuracy']:.4f}")
        fig, ax = plt.subplots(figsize=(4, 3))
        plot_confusion_matrix_ax(row["confusion_matrix"], ax, title=f"{row['name']} [{row['voting']}]")


def plot_confusion_matrix_ax(cm, ax, title=None):
    """Render a confusion matrix `cm` on axes `ax` with labels and colorbar."""
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title or "Confusion matrix")
    classes = np.arange(cm.shape[0])
    ax.set_xticks(classes)
    ax.set_yticks(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # annotate cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def analyze_experiments(committee_results, y_test, X_test):
    print("\n" + "=" * 80)
    print("EXPERIMENT ANALYSIS")
    print("=" * 80)

    grouped = {}
    for row in committee_results:
        grouped.setdefault((row["kind"], row["voting"]), []).append((row["n_members"], row["accuracy"]))

    print("\nEffect of increasing number of classifiers:")
    for (committee_type, voting), values in sorted(grouped.items()):
        values = sorted(values, key=lambda x: x[0])
        trend = " -> ".join([f"N={n}:{acc:.4f}" for n, acc in values])
        print(f"- {committee_type} [{voting}] : {trend}")

    print("\nCases where majority voting fails (majority wrong, weighted correct):")
    seen = set()
    for row in committee_results:
        key = row["name"]
        if row["n_members"] != 7 or key in seen:
            continue

        committee = row["committee"]
        majority_pred = committee.predict_majority(X_test)
        weighted_pred = committee.predict_weighted(X_test)

        fail_count = np.sum((majority_pred != y_test) & (weighted_pred == y_test))
        weighted_fail_count = np.sum((weighted_pred != y_test) & (majority_pred == y_test))

        print(
            f"- {key}: majority_failed_weighted_fixed={fail_count}, "
            f"weighted_failed_majority_fixed={weighted_fail_count}"
        )
        seen.add(key)


def plot_dataset(X, y):
    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=24, edgecolor="k", alpha=0.75)
    plt.title("Synthetic Non-Linear Dataset (Two Slightly Overlapping Classes)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()


def plot_decision_boundary(ax, predictor, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 260),
        np.linspace(y_min, y_max, 260),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = predictor(grid).reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.30, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=10, edgecolor="k", alpha=0.75)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_comparison(X, y, single_results, committee_results):
    best_single = max(single_results, key=lambda r: r["test_accuracy"])

    weighted_rows = [r for r in committee_results if r["voting"] == "weighted"]
    best_bagging = max(
        [r for r in weighted_rows if r["name"].startswith("Bagging")],
        key=lambda r: r["accuracy"],
    )
    best_diverse = max(
        [r for r in weighted_rows if r["name"].startswith("Diverse classifiers")],
        key=lambda r: r["accuracy"],
    )
    best_param = max(
        [r for r in weighted_rows if r["name"].startswith("Same classifier, different params")],
        key=lambda r: r["accuracy"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_decision_boundary(
        axes[0, 0],
        best_single["model"].predict,
        X,
        y,
        f"Best Single: {best_single['name']} ({best_single['test_accuracy']:.3f})",
    )
    plot_decision_boundary(
        axes[0, 1],
        best_bagging["committee"].predict_weighted,
        X,
        y,
        f"Best Bagging Committee ({best_bagging['accuracy']:.3f})",
    )
    plot_decision_boundary(
        axes[1, 0],
        best_diverse["committee"].predict_weighted,
        X,
        y,
        f"Best Diverse Committee ({best_diverse['accuracy']:.3f})",
    )
    plot_decision_boundary(
        axes[1, 1],
        best_param["committee"].predict_weighted,
        X,
        y,
        f"Best Same-Classifier Param Committee ({best_param['accuracy']:.3f})",
    )

    plt.suptitle("Decision Boundary Comparison: Single Classifier vs Committees", fontsize=14)
    plt.tight_layout()


def textual_summary(single_results, committee_results):
    best_single = max(single_results, key=lambda r: r["test_accuracy"])
    best_committee = max(committee_results, key=lambda r: r["accuracy"])

    print("\n" + "=" * 80)
    print("SHORT SUMMARY")
    print("=" * 80)
    print(
        f"Best single classifier: {best_single['name']} "
        f"(test accuracy={best_single['test_accuracy']:.4f})"
    )
    print(
        f"Best committee: {best_committee['name']} with {best_committee['voting']} voting "
        f"(test accuracy={best_committee['accuracy']:.4f})"
    )

    improvement = best_committee["accuracy"] - best_single["test_accuracy"]
    if improvement > 0:
        print(f"Committee improves over best single model by +{improvement:.4f} accuracy.")
    elif improvement < 0:
        print(f"Best single model is better by {abs(improvement):.4f} accuracy.")
    else:
        print("Best committee and best single model achieve equal test accuracy.")

    print(
        "Weighted voting often helps when stronger members disagree with weaker ones, "
        "while plain majority voting can fail in those cases."
    )


def main():
    np.random.seed(42)

    X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test = generate_data(random_state=42)

    candidates = get_base_candidates(random_state=42)
    best_models = []
    for name, model_candidates in candidates.items():
        best_models.append(tune_model(name, model_candidates, X_train, y_train, X_val, y_val))

    single_results = evaluate_single_classifiers(best_models, X_test, y_test)

    n_values = [3, 5, 7]

    best_tree = next(m for n, m, _ in best_models if n == "Decision Tree")
    bagging_committees = build_bagging_committees(
        n_values=n_values,
        base_model=best_tree,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seed=42,
    )

    diverse_pool = build_diverse_model_pool(best_models, random_state=42)
    diverse_committees = build_diverse_committees(
        n_values=n_values,
        model_pool=diverse_pool,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    knn_param_committees = build_knn_param_committees(
        n_values=n_values,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    all_committees = bagging_committees + diverse_committees + knn_param_committees
    committee_results = evaluate_committees(all_committees, X_test, y_test)

    print_results(single_results, committee_results)
    analyze_experiments(committee_results, y_test, X_test)
    textual_summary(single_results, committee_results)

    plot_dataset(X_all, y_all)
    plot_comparison(X_all, y_all, single_results, committee_results)
    plt.show()


if __name__ == "__main__":
    main()
