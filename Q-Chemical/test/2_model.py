import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch")
def quantum_layer(inputs, weights):
    for i in range(4):
        qml.RY(inputs[i % len(inputs)], wires=i)
    for i in range(3):
        qml.CNOT(wires=[i, i+1])
    for i in range(4):
        qml.RZ(weights[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = nn.Parameter(0.01 * torch.randn(4))
        self.fc = nn.Linear(4, 10)
    def forward(self, x):
        q_out = torch.stack([torch.tensor(quantum_layer(xi, self.q_weights)) for xi in x])
        return self.fc(q_out.float())

print("Hybrid Quantum Model Architecture Defined.")
