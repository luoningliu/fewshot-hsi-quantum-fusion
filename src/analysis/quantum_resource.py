from __future__ import annotations


def estimate_qnn_resources(qubits: int, layers: int, entanglement: str) -> dict[str, int | str]:
    if entanglement == "none":
        entangling_gates_per_layer = 0
    elif entanglement == "linear":
        entangling_gates_per_layer = max(qubits - 1, 0)
    elif entanglement == "ring":
        entangling_gates_per_layer = qubits if qubits > 2 else max(qubits - 1, 0)
    elif entanglement == "full":
        entangling_gates_per_layer = qubits * (qubits - 1) // 2
    else:
        raise ValueError(f"Unsupported entanglement: {entanglement}")
    return {
        "qubits": qubits,
        "layers": layers,
        "entanglement": entanglement,
        "single_qubit_rotations": qubits + 3 * qubits * layers,
        "entangling_gates": entangling_gates_per_layer * layers,
    }

