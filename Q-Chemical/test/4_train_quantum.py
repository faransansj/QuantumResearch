import sys
try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata
sys.modules["importlib.metadata"] = metadata

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from rdkit import Chem
import pennylane as qml

# 1. 데이터 로드
df = pd.read_csv('processed_data.csv')
smiles_list = df['smiles'].tolist()
tokens = sorted(list(set("".join(smiles_list))))
token_to_int = {t: i for i, t in enumerate(tokens)}

# 2. 양자 모델 정의
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch")
def q_layer(inputs, weights):
    for i in range(4): qml.RY(inputs[0], wires=i) # 각 문자 각도 입력
    for i in range(3): qml.CNOT(wires=[i, i+1])
    for i in range(4): qml.RZ(weights[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class SimpleQuantumModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_weights = torch.nn.Parameter(0.01 * torch.randn(4))
        self.fc = torch.nn.Linear(4, len(tokens))

    def forward(self, x):
        # x shape: (1, seq_len)
        seq_len = x.shape[1]
        results = []
        # 각 문자의 각도를 순차적으로 양자 회로에 입력
        for i in range(seq_len):
            char_angle = x[0, i:i+1] 
            q_val = q_layer(char_angle, self.q_weights)
            results.append(q_val)
        
        q_out = torch.stack(results) # (seq_len, 4)
        return self.fc(q_out.float()) # (seq_len, len(tokens))

# 3. 학습 루프
print("\n--- 양자 AI 사전 학습 시작 (글자 단위 학습 중) ---")
model = SimpleQuantumModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    total_loss = 0
    # 데이터셋의 처음 5개 분자로 학습 진행
    for s in smiles_list[:5]:
        input_angles = torch.tensor([(token_to_int[c]/len(tokens))*2*np.pi for c in s]).unsqueeze(0)
        target = torch.tensor([token_to_int[c] for c in s])
        
        optimizer.zero_grad()
        output = model(input_angles) # (seq_len, 13)
        
        # 이제 출력(34, 13)과 타겟(34)의 차원이 일치합니다.
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10 | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), 'quantum_model.ckpt')
print("\n학습 완료! 모델이 'quantum_model.ckpt'에 저장되었습니다.")