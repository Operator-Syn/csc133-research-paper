import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import CONFIG, LABEL_MAP


def evaluate_classifier(model, X_test, y_test, title="Confusion Matrix"):
    """
    Evaluate a trained classifier using accuracy, a classification report,
    and a confusion matrix.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[LABEL_MAP[label] for label in sorted(LABEL_MAP.keys())],
        )
    )

    labels = sorted(LABEL_MAP.keys())
    label_names = [LABEL_MAP[label] for label in labels]

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    figure_dir = CONFIG.results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    safe_title = title.lower().replace(" ", "_")
    plt.savefig(
        figure_dir / f"{safe_title}.svg",
        format="svg",
        bbox_inches="tight",
    )

    plt.show()

    return {
        "accuracy": accuracy,
        "y_pred": y_pred,
        "confusion_matrix": matrix,
    }
