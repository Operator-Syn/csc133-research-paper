from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config import CONFIG, LABEL_MAP, TARGET_LABELS


def fetch_intent_data(
    labels: list[int] | None = None,
    samples_per_label: int | None = None,
    test_size: float | None = None,
    random_state: int | None = None,
    balance_classes: bool | None = None,
):
    """
    Load the Tanaos synthetic intent classifier dataset and prepare the
    Greeting/Farewell subset for binary intent classification.

    Parameters default to values defined in src.config.CONFIG.
    """

    labels = TARGET_LABELS if labels is None else labels
    samples_per_label = CONFIG.samples_per_label if samples_per_label is None else samples_per_label
    test_size = CONFIG.test_size if test_size is None else test_size
    random_state = CONFIG.base_seed if random_state is None else random_state
    balance_classes = CONFIG.balance_classes if balance_classes is None else balance_classes

    print(f"Requesting dataset: tanaos/synthetic-intent-classifier-dataset-v1")

    raw_dataset = load_dataset(
        "tanaos/synthetic-intent-classifier-dataset-v1",
        split="train",
    )

    df = raw_dataset.to_pandas()

    # Standardize possible label column names.
    if "intent" in df.columns:
        df = df.rename(columns={"intent": "label"})
    elif "labels" in df.columns:
        df = df.rename(columns={"labels": "label"})

    if "label" not in df.columns:
        raise KeyError(
            f"Could not find a label column. Available columns: {list(df.columns)}"
        )

    if CONFIG.text_column not in df.columns:
        raise KeyError(
            f"Could not find text column '{CONFIG.text_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only the target labels from the research configuration.
    df_filtered = df[df["label"].isin(labels)].copy()

    if df_filtered.empty:
        raise ValueError(f"No rows found for target labels: {labels}")

    # Original class counts before balancing.
    original_counts = (
        df_filtered["label"]
        .value_counts()
        .sort_index()
        .rename(index=LABEL_MAP)
    )

    # Balance the selected classes if enabled.
    if balance_classes:
        balanced_frames = []

        for label, group in df_filtered.groupby("label"):
            n_samples = min(len(group), samples_per_label)

            sampled_group = group.sample(
                n=n_samples,
                random_state=random_state,
            )

            balanced_frames.append(sampled_group)

        final_df = pd.concat(balanced_frames).reset_index(drop=True)
    else:
        final_df = df_filtered.reset_index(drop=True)

    # Final class counts after optional balancing.
    final_counts = (
        final_df["label"]
        .value_counts()
        .sort_index()
        .rename(index=LABEL_MAP)
    )

    train_df, test_df = train_test_split(
        final_df,
        test_size=test_size,
        stratify=final_df["label"],
        random_state=random_state,
    )

    train_counts = (
        train_df["label"]
        .value_counts()
        .sort_index()
        .rename(index=LABEL_MAP)
    )

    test_counts = (
        test_df["label"]
        .value_counts()
        .sort_index()
        .rename(index=LABEL_MAP)
    )

    summary = {
        "original_counts": original_counts,
        "final_counts": final_counts,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "total_samples": len(final_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "test_size": test_size,
        "labels": labels,
        "balance_classes": balance_classes,
        "samples_per_label": samples_per_label,
        "random_state": random_state,
    }

    print(f"Success: {len(train_df)} training and {len(test_df)} testing samples retrieved.")

    return train_df, test_df, summary
