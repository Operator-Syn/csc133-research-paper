from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from src.config import CONFIG


def build_depolarizing_noise_model(noise_level: float) -> NoiseModel | None:
    """
    Build a depolarizing noise model for simulation.

    A noise level of 0.0 returns None, which represents the noiseless simulator.
    For nonzero noise levels, single-qubit and two-qubit depolarizing errors
    are attached to common circuit gates.
    """

    if noise_level <= 0.0:
        return None

    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be between 0.0 and 1.0")

    noise_model = NoiseModel()

    single_qubit_error = depolarizing_error(noise_level, 1)
    two_qubit_error = depolarizing_error(noise_level, 2)

    # Common one-qubit gates after circuit decomposition/transpilation.
    one_qubit_gates = [
        "id",
        "x",
        "sx",
        "h",
        "rx",
        "ry",
        "rz",
    ]

    # Common two-qubit gates used after decomposition/transpilation.
    two_qubit_gates = [
        "cx",
        "cz",
    ]

    noise_model.add_all_qubit_quantum_error(single_qubit_error, one_qubit_gates)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, two_qubit_gates)

    return noise_model


def run_noisy_counts(
    circuit: QuantumCircuit,
    noise_level: float,
    shots: int,
    seed: int | None = None,
) -> dict:
    """
    Run a quantum circuit using an Aer simulator with optional depolarizing noise.

    The circuit is measured before execution. The function returns raw measurement
    counts such as {'00': 512, '11': 488}.
    """

    measured_circuit = circuit.copy()
    measured_circuit.measure_all()

    noise_model = build_depolarizing_noise_model(noise_level)

    simulator = AerSimulator(
        noise_model=noise_model,
        seed_simulator=CONFIG.base_seed if seed is None else seed,
    )

    transpiled_circuit = transpile(
        measured_circuit,
        simulator,
        seed_transpiler=CONFIG.base_seed if seed is None else seed,
    )

    result = simulator.run(
        transpiled_circuit,
        shots=shots,
    ).result()

    return result.get_counts()
