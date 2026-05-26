from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from sklearn.metrics import accuracy_score

from src.noise_simulation import build_depolarizing_noise_model


@dataclass(frozen=True)
class FixedNoisyRunConfig:
    """
    Configuration for one fixed-model noisy evaluation run.
    """

    noise_level: float
    shots: int
    seed: int


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
        "Make sure the VQC has been fitted before noisy evaluation."
    )


def counts_to_binary_label(counts: dict[str, int]) -> int:
    """
    Convert measured bitstring counts into a binary class label.

    The same parity-style interpretation used by Qiskit's VQC default binary
    interpretation is applied here: even parity maps to class 0, odd parity maps
    to class 1.
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
    run_config: FixedNoisyRunConfig,
) -> dict[str, Any]:
    """
    Evaluate a trained VQC under depolarizing noise and finite shots.

    The VQC is not retrained. The trained parameters are reused, and only the
    simulator noise level, shot count, and seed are changed.
    """

    trained_weights = get_trained_weights(vqc_model)

    full_circuit = feature_map.compose(ansatz)
    full_circuit.measure_all()
    full_circuit = full_circuit.decompose()

    noise_model = build_depolarizing_noise_model(run_config.noise_level)

    backend = AerSimulator(
        noise_model=noise_model,
        seed_simulator=run_config.seed,
    )

    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
        seed_transpiler=run_config.seed,
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
        shots=run_config.shots,
    ).result()

    predictions = []

    for index in range(len(bound_circuits)):
        counts = result.get_counts(index)
        prediction = counts_to_binary_label(counts)
        predictions.append(prediction)

    predictions = np.asarray(predictions)
    accuracy = accuracy_score(y_test, predictions)

    output = asdict(run_config)
    output["accuracy"] = float(accuracy)

    return output


def run_fixed_noisy_sweep(
    jobs: list[FixedNoisyRunConfig],
    vqc_model,
    feature_map,
    ansatz,
    X_test,
    y_test,
) -> list[dict[str, Any]]:
    """
    Run a fixed-model noisy evaluation sweep.

    Each job evaluates the same trained VQC under a different noise and shot
    configuration.
    """

    rows = []

    print(f"Running {len(jobs)} fixed-model noisy evaluation(s)...")

    for index, job in enumerate(jobs, start=1):
        print(
            f"\n[{index}/{len(jobs)}] "
            f"noise={job.noise_level:.2f}, shots={job.shots}, seed={job.seed}"
        )

        result = evaluate_fixed_vqc_under_noise(
            vqc_model=vqc_model,
            feature_map=feature_map,
            ansatz=ansatz,
            X_test=X_test,
            y_test=y_test,
            run_config=job,
        )

        rows.append(result)

        print(
            f"done | noise={result['noise_level']:.2f}, "
            f"shots={result['shots']}, "
            f"seed={result['seed']}, "
            f"accuracy={result['accuracy']:.4f}"
        )

    return rows
