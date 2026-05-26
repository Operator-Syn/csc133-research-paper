
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Central configuration for the VQC intent classification experiment.

    This keeps the main experiment settings in one place so the notebook and
    source files can reuse the same labels, seeds, feature dimensions, noise
    levels, shot counts, and trial counts.
    """

    # Project paths
    project_root: Path = Path.cwd()
    data_dir: Path = Path.cwd() / "data"
    results_dir: Path = Path.cwd() / "results"

    # Dataset settings
    dataset_name: str = "tanaos/synthetic-intent-classifier-dataset-v1"
    text_column: str = "text"
    label_column: str = "label"

    greeting_label: int = 0
    farewell_label: int = 1

    greeting_name: str = "Greeting"
    farewell_name: str = "Farewell"

    samples_per_label: int = 500
    test_size: float = 0.20
    balance_classes: bool = True

    # Reproducibility
    base_seed: int = 42

    # Feature extraction
    tfidf_max_features: int = 500
    pca_components: int = 2

    # VQC architecture
    num_qubits: int = 2
    feature_map: str = "ZZFeatureMap"
    feature_map_reps: int = 1
    ansatz: str = "RealAmplitudes"
    ansatz_reps: int = 2
    trainable_parameters: int = 6
    optimizer: str = "COBYLA"
    maxiter: int = 100

    # Noise and sampling settings
    noise_levels: tuple[float, ...] = (0.00, 0.01, 0.03, 0.05, 0.07, 0.10)
    shot_counts: tuple[int, ...] = (100, 1000, 10000)

    # Zero-Noise Extrapolation settings
    zne_scale_factors: tuple[int, ...] = (1, 3, 5)

    # Monte Carlo settings
    monte_carlo_trials: int = 1000


CONFIG = ExperimentConfig()

LABEL_MAP = {
    CONFIG.greeting_label: CONFIG.greeting_name,
    CONFIG.farewell_label: CONFIG.farewell_name,
}

TARGET_LABELS = list(LABEL_MAP.keys())
