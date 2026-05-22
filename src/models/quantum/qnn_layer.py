from __future__ import annotations

import torch
from torch import nn

try:
    import pennylane as qml
except Exception:  # pragma: no cover - surfaced at runtime when QNN is instantiated.
    qml = None


class PennyLaneQNNLayer(nn.Module):
    def __init__(
        self,
        qubits: int = 6,
        layers: int = 2,
        entanglement: str = "ring",
        trainable: bool = True,
        backend: str = "lightning.qubit",
        diff_method: str = "adjoint",
    ):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required for PennyLaneQNNLayer.")
        self.qubits = qubits
        self.layers = layers
        self.entanglement = entanglement
        self.backend = backend
        self.diff_method = diff_method
        self.weights = nn.Parameter(0.01 * torch.randn(layers, qubits, 3), requires_grad=trainable)
        self.dev = qml.device(backend, wires=qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method=diff_method)

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor):
        for wire in range(self.qubits):
            qml.RY(inputs[wire], wires=wire)
        for layer in range(self.layers):
            for wire in range(self.qubits):
                qml.Rot(weights[layer, wire, 0], weights[layer, wire, 1], weights[layer, wire, 2], wires=wire)
            _apply_entanglement(self.qubits, self.entanglement)
        return [qml.expval(qml.PauliZ(wire)) for wire in range(self.qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.qubits:
            raise ValueError(f"Expected last dimension {self.qubits}, got {x.shape[-1]}")
        outputs = [torch.stack(self.qnode(sample, self.weights)) for sample in x]
        return torch.stack(outputs, dim=0).to(dtype=x.dtype)


def _apply_entanglement(qubits: int, entanglement: str) -> None:
    if entanglement == "none":
        return
    if entanglement == "linear":
        for wire in range(qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])
        return
    if entanglement == "ring":
        for wire in range(qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])
        if qubits > 2:
            qml.CNOT(wires=[qubits - 1, 0])
        return
    if entanglement == "full":
        for control in range(qubits):
            for target in range(control + 1, qubits):
                qml.CNOT(wires=[control, target])
        return
    raise ValueError(f"Unsupported entanglement: {entanglement}")
