from __future__ import annotations

import math

import torch
from torch import nn

try:
    import pennylane as qml
except Exception:  # pragma: no cover
    qml = None

from src.models.quantum.qnn_layer import _apply_entanglement


class DataReuploadingMultiObservableQNNLayer(nn.Module):
    def __init__(
        self,
        qubits: int = 6,
        layers: int = 2,
        entanglement: str = "linear",
        backend: str = "lightning.qubit",
        diff_method: str = "adjoint",
    ):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required for DataReuploadingMultiObservableQNNLayer.")
        self.qubits = qubits
        self.layers = layers
        self.entanglement = entanglement
        self.output_dim = qubits * 2
        self.weights = nn.Parameter(0.01 * torch.randn(layers, qubits, 3))
        self.input_scales = nn.Parameter(torch.ones(layers, qubits))
        self.input_shifts = nn.Parameter(torch.zeros(layers, qubits))
        self.dev = qml.device(backend, wires=qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method=diff_method)

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor, scales: torch.Tensor, shifts: torch.Tensor):
        for layer in range(self.layers):
            encoded = scales[layer] * inputs + shifts[layer]
            for wire in range(self.qubits):
                qml.RY(encoded[wire], wires=wire)
            for wire in range(self.qubits):
                qml.Rot(weights[layer, wire, 0], weights[layer, wire, 1], weights[layer, wire, 2], wires=wire)
            _apply_entanglement(self.qubits, self.entanglement)

        observables = [qml.expval(qml.PauliZ(wire)) for wire in range(self.qubits)]
        observables.extend(
            qml.expval(qml.PauliZ(wire) @ qml.PauliZ((wire + 1) % self.qubits)) for wire in range(self.qubits)
        )
        return observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.qubits:
            raise ValueError(f"Expected last dimension {self.qubits}, got {x.shape[-1]}")
        outputs = [torch.stack(self.qnode(sample, self.weights, self.input_scales, self.input_shifts)) for sample in x]
        return torch.stack(outputs, dim=0).to(dtype=x.dtype)


class DataReuploadingMultiObservableQNNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        qubits: int = 6,
        layers: int = 2,
        entanglement: str = "linear",
        backend: str = "lightning.qubit",
        diff_method: str = "adjoint",
        normalize_input: bool = True,
        angle_scale: float = math.pi,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if normalize_input else nn.Identity()
        self.project = nn.Linear(input_dim, qubits)
        self.angle_scale = float(angle_scale)
        self.qnn = DataReuploadingMultiObservableQNNLayer(
            qubits=qubits,
            layers=layers,
            entanglement=entanglement,
            backend=backend,
            diff_method=diff_method,
        )
        self.classifier = nn.Linear(self.qnn.output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.angle_scale * torch.tanh(self.project(x))
        x = self.qnn(x)
        return self.classifier(x)


class ResidualDataReuploadingMultiObservableQNNClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, **qnn_kwargs):
        super().__init__()
        self.linear_head = nn.Linear(input_dim, num_classes)
        self.qnn_head = DataReuploadingMultiObservableQNNClassifier(input_dim, num_classes, **qnn_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_head(x) + self.qnn_head(x)

