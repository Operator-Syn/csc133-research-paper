from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA

from src.config import CONFIG


def parity(bitstring_as_int: int) -> int:
    """
    Map a measured computational-basis state to a binary class using parity.

    Even parity maps to class 0.
    Odd parity maps to class 1.

    This keeps the clean VQC and the noisy/ZNE evaluator aligned.
    """

    return bin(bitstring_as_int).count("1") % 2


def build_vqc():
    """
    Build the clean baseline Variational Quantum Classifier.

    This model uses StatevectorSampler, so it does not include depolarizing
    noise, finite-shot sampling, or error mitigation. The trained parameters
    from this clean model are reused later for fixed-model noisy evaluation.

    The VQC uses an explicit parity interpretation so that its class mapping
    matches the manual noisy/ZNE evaluator.
    """

    feature_map = ZZFeatureMap(
        feature_dimension=CONFIG.pca_components,
        reps=CONFIG.feature_map_reps,
    )

    ansatz = RealAmplitudes(
        num_qubits=CONFIG.num_qubits,
        reps=CONFIG.ansatz_reps,
    )

    optimizer = COBYLA(
        maxiter=CONFIG.maxiter,
    )

    sampler = StatevectorSampler()

    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        interpret=parity,
        output_shape=2,
    )

    return vqc, feature_map, ansatz


def train_vqc(X_train, y_train):
    """
    Train the clean baseline VQC using the processed training features.
    """

    vqc, feature_map, ansatz = build_vqc()

    print("Building clean baseline VQC...")
    print(f"Feature map  : {CONFIG.feature_map}")
    print(f"Feature reps : {CONFIG.feature_map_reps}")
    print(f"Ansatz       : {CONFIG.ansatz}")
    print(f"Ansatz reps  : {CONFIG.ansatz_reps}")
    print(f"Qubits       : {CONFIG.num_qubits}")
    print(f"Parameters   : {len(ansatz.parameters)}")
    print(f"Optimizer    : {CONFIG.optimizer}")
    print(f"Max iter     : {CONFIG.maxiter}")
    print("Interpret    : parity")
    print("Output shape : 2")

    print("\nTraining clean baseline VQC...")
    vqc.fit(X_train, y_train)

    print("Clean baseline VQC training complete.")

    return vqc, feature_map, ansatz
