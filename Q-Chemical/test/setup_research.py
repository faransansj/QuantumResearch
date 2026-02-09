import os

# 1. 기초 데이터 생성 (oled_db.txt) [cite: 206, 207]
oled_db_content = """Cc1ccc(-c2ccc(-c3ccc(C)cc3)cc2)cc1
c1ccc(N(c2ccccc2)c2ccc(-c3ccc4ccccc4c3)cc2)cc1
C1=CC=C(C=C1)C1=CC2=C(C=C1)C1=CC=C(C=C1)N2C1=CC=C(C=C1)C1=CC=CC=C1
O=C1c2ccccc2-c2ccccc21
c1ccc(-c2c3ccccc3c(-c3ccccc3)c3ccccc23)cc1
"""

# 2. 데이터 증강 코드 (0_augmentation.py) [cite: 10, 164, 165]
augmentation_code = """from rdkit import Chem
import os

def augment_smiles(input_file, aug_factor=5):
    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]
    augmented = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            new_set = {s}
            attempts = 0
            while len(new_set) < aug_factor and attempts < 50:
                new_set.add(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
                attempts += 1
            augmented.extend(list(new_set))
    with open('aug_data.txt', 'w') as f:
        for s in augmented:
            f.write(s + '\\n')
    print(f"DONE: {len(augmented)} smiles saved to aug_data.txt")

if __name__ == "__main__":
    augment_smiles('oled_db.txt', aug_factor=5)
"""

# 3. 데이터 정제 및 양자 매핑 코드 (1_processing.py) [cite: 32, 169, 171]
processing_code = """import pandas as pd
from rdkit import Chem
import numpy as np

def process_data(input_file):
    with open(input_file, 'r') as f:
        smiles = [line.strip() for line in f.readlines()]
    valid = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            valid.append(Chem.MolToSmiles(mol, canonical=True))
    tokens = sorted(list(set("".join(valid))))
    token_to_int = {t: i for i, t in enumerate(tokens)}
    processed = []
    for s in valid:
        angles = [(token_to_int[char] / len(tokens)) * 2 * np.pi for char in s]
        processed.append({'smiles': s, 'angles': angles})
    pd.DataFrame(processed).to_csv('processed_data.csv', index=False)
    print(f"DONE: Final data saved. Tokens: {tokens}")

if __name__ == "__main__":
    process_data('aug_data.txt')
"""

# 4. 양자 하이브리드 모델 정의 (2_model.py) [cite: 43, 173, 174]
model_code = """import pennylane as qml
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
"""

# 파일 쓰기 실행
files = {
    "oled_db.txt": oled_db_content,
    "0_augmentation.py": augmentation_code,
    "1_processing.py": processing_code,
    "2_model.py": model_code
}

for filename, content in files.items():
    with open(filename, 'w') as f:
        f.write(content)
    print(f"Created: {filename}")

print("\\n[SUCCESS] 모든 연구 파일이 생성되었습니다. 이제 순서대로 실행하세요.")