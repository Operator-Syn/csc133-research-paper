from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from sklearn.metrics import accuracy_score

from src.noise_simulation import build_depolarizing_noise_model


@dataclass(frozen=True)
class ZNERunConfig:
    """
    Configuration for one fixed-model ZNE evaluation run.
    """

    base_noise_level: float
    shots: int
    seed: int
    scale_factors: tuple[int, ...]


def get_trained_weights(vqc_model) -> np.ndarray:
    """
    Extract trained weights from a fitted Qiskit Machine Learning VQC.
    """

    if hasattr(vqc_model, "weights") and vqc_model.weights is not None:
        return np.asarray(vqc_model.weights, dtype=float)

    if hasattr(vqc_model, "fit_result") and vqc_model.fit_result is not None:
        return np.asarray(vqc_model.fit_result.x, dtype=float)

    if hasattr(vqc_model, "_fit_result") and vqc_model._fit_result is not None:
        return np.asarray(vqc_model._fit_result.x, dtype=float)

    raise AttributeError(
        "Could not extract trained weights from the VQC model. "
        "Make sure the VQC has been fitted before noisy or ZNE evaluation."
    )


def counts_to_binary_label(counts: dict[str, int]) -> int:
    """
    Convert measured bitstring counts into a binary class label.

    The same parity-style interpretation used by Qiskit's default binary VQC
    interpretation is applied here: even parity maps to class 0, odd parity
    maps to class 1.
    """

    class_counts = {
        0: 0,
        1: 0,
    }

    for bitstring, count in counts.items():
        cleaned = bitstring.replace(" ", "")
        parity = sum(int(bit) for bit in cleaned) % 2
        class_counts[parity] += count

    if class_counts[0] >= class_counts[1]:
        return 0

    return 1


def build_bound_circuits(
    parameterized_circuit,
    feature_map,
    ansatz,
    trained_weights: np.ndarray,
    X,
):
    """
    Bind test features and trained VQC weights to the parameterized circuit.

    The VQC is not retrained. Each test sample receives its own feature values,
    while all samples reuse the same trained ansatz weights.
    """

    input_parameters = list(feature_map.parameters)
    weight_parameters = list(ansatz.parameters)

    if len(input_parameters) != X.shape[1]:
        raise ValueError(
            f"Expected {len(input_parameters)} input features, "
            f"but received {X.shape[1]}."
        )

    if len(weight_parameters) != len(trained_weights):
        raise ValueError(
            f"Expected {len(weight_parameters)} trained weights, "
            f"but received {len(trained_weights)}."
        )

    bound_circuits = []

    for sample in X:
        parameter_values = {}

        for parameter, value in zip(input_parameters, sample):
            parameter_values[parameter] = float(value)

        for parameter, value in zip(weight_parameters, trained_weights):
            parameter_values[parameter] = float(value)

        bound_circuit = parameterized_circuit.assign_parameters(
            parameter_values,
            inplace=False,
        )

        bound_circuits.append(bound_circuit)

    return bound_circuits


def evaluate_fixed_vqc_under_noise(
    vqc_model,
    feature_map,
    ansatz,
    X_test,
    y_test,
    noise_level: float,
    shots: int,
    seed: int,
) -> float:
    """
    Evaluate a trained VQC under depolarizing noise and finite shots.

    The VQC is not retrained. The trained parameters are reused, and only the
    simulator noise level, shot count, and seed are changed.
    """

    trained_weights = get_trained_weights(vqc_model)

    full_circuit = feature_map.compose(ansatz)
    full_circuit.measure_all()
    full_circuit = full_circuit.decompose()

    noise_model = build_depolarizing_noise_model(noise_level)

    backend = AerSimulator(
        noise_model=noise_model,
        seed_simulator=seed,
    )

    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
        seed_transpiler=seed,
    )

    transpiled_circuit = pass_manager.run(full_circuit)

    bound_circuits = build_bound_circuits(
        parameterized_circuit=transpiled_circuit,
        feature_map=feature_map,
        ansatz=ansatz,
        trained_weights=trained_weights,
        X=X_test,
    )

    result = backend.run(
        bound_circuits,
        shots=shots,
    ).result()

    predictions = []

    for index in range(len(bound_circuits)):
        counts = result.get_counts(index)
        prediction = counts_to_binary_label(counts)
        predictions.append(prediction)

    predictions = np.asarray(predictions)
    accuracy = accuracy_score(y_test, predictions)

    return float(accuracy)


def extrapolate_zero_noise(
    scaled_noise_levels: list[float],
    scaled_accuracies: list[float],
) -> float:
    """
    Estimate the zero-noise accuracy using linear extrapolation.

    The x-axis is the scaled noise level.
    The y-axis is the observed accuracy at that scaled noise level.
    The extrapolated value is the fitted intercept at noise = 0.
    """

    x = np.asarray(scaled_noise_levels, dtype=float)
    y = np.asarray(scaled_accuracies, dtype=float)

    if len(set(x)) == 1:
        return float(y[0])

    slope, intercept = np.polyfit(x, y, deg=1)

    return float(np.clip(intercept, 0.0, 1.0))


def evaluate_zne_fixed_vqc(
    vqc_model,
    feature_map,
    ansatz,
    X_test,
    y_test,
    run_config: ZNERunConfig,
) -> dict[str, Any]:
    """
    Evaluate a fixed trained VQC using Zero-Noise Extrapolation.

    The trained VQC is not retrained. The same trained parameters are evaluated
    at scaled depolarizing noise levels, then linearly extrapolated to estimate
    the zero-noise accuracy.
    """

    scaled_noise_levels = []
    scaled_accuracies = []

    for scale_factor in run_config.scale_factors:
        scaled_noise_level = run_config.base_noise_level * scale_factor
        scaled_noise_level = min(scaled_noise_level, 1.0)

        accuracy = evaluate_fixed_vqc_under_noise(
            vqc_model=vqc_model,
            feature_map=feature_map,
            ansatz=ansatz,
            X_test=X_test,
            y_test=y_test,
            noise_level=scaled_noise_level,
            shots=run_config.shots,
            seed=run_config.seed,
        )

        scaled_noise_levels.append(scaled_noise_level)
        scaled_accuracies.append(accuracy)

    raw_accuracy = scaled_accuracies[0]

    zne_accuracy = extrapolate_zero_noise(
        scaled_noise_levels=scaled_noise_levels,
        scaled_accuracies=scaled_accuracies,
    )

    output = asdict(run_config)
    output["raw_accuracy"] = float(raw_accuracy)
    output["zne_accuracy"] = float(zne_accuracy)
    output["zne_delta"] = float(zne_accuracy - raw_accuracy)
    output["scaled_noise_levels"] = tuple(scaled_noise_levels)
    output["scaled_accuracies"] = tuple(scaled_accuracies)

    return output


def run_zne_sweep(
    jobs: list[ZNERunConfig],
    vqc_model,
    feature_map,
    ansatz,
    X_test,
    y_test,
) -> list[dict[str, Any]]:
    """
    Run fixed-model ZNE evaluation across all configured jobs.
    """

    rows = []

    print(f"Running {len(jobs)} ZNE evaluation(s)...")

    for index, job in enumerate(jobs, start=1):
        print(
            f"\n[{index}/{len(jobs)}] "
            f"base_noise={job.base_noise_level:.2f}, "
            f"shots={job.shots}, "
            f"seed={job.seed}, "
            f"scales={job.scale_factors}"
        )

        result = evaluate_zne_fixed_vqc(
            vqc_model=vqc_model,
            feature_map=feature_map,
            ansatz=ansatz,
            X_test=X_test,
            y_test=y_test,
            run_config=job,
        )

        rows.append(result)

        print(
            f"done | base_noise={result['base_noise_level']:.2f}, "
            f"shots={result['shots']}, "
            f"seed={result['seed']}, "
            f"raw={result['raw_accuracy']:.4f}, "
            f"zne={result['zne_accuracy']:.4f}, "
            f"delta={result['zne_delta']:.4f}"
        )

    return rows
