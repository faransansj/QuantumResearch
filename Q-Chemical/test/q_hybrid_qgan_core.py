import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

def create_quantum_generator():
    print("\n--- [고도화] Qiskit 기반 Hybrid QGAN 생성자 설계 ---")
    
    # 1. 인코딩 레이어 (ZZFeatureMap): 잠재 공간(Latent Space) 투사
    # 노이즈 데이터를 양자 상태로 변환합니다.
    feature_map = ZZFeatureMap(feature_dimension=4, reps=1, entanglement='linear')
    
    # 2. 가변 양자 회로 (Ansatz): 학습 가능한 레이어
    # RealAmplitudes는 생성 모델에서 가장 전문적으로 쓰이는 구조 중 하나입니다.
    ansatz = RealAmplitudes(num_qubits=4, reps=3, entanglement='full')
    
    # 3. 전체 회로 구성
    qc = QuantumCircuit(4)
    qc.append(feature_map, range(4))
    qc.append(ansatz, range(4))
    
    # 4. Qiskit QNN 정의: 양자 회로를 신경망 계층으로 변환
    # SamplerQNN은 측정 확률(Probability)을 출력으로 내보냅니다.
    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters
    )
    
    # 5. PyTorch 연결 (TorchConnector): 전문 하이브리드 통합
    # 이제 이 양자 회로는 일반적인 PyTorch 레이어처럼 동작합니다.
    return TorchConnector(qnn)

class QuantumGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnn = create_quantum_generator()
        # 양자 출력(확률)을 분자 토큰 크기로 매핑하는 최종 선형층
        self.fc = nn.Linear(16, 13) # 2^4=16 (큐비트 상태 수), 13 (SMILES 토큰 수)

    def forward(self, x):
        # x는 잠재 공간의 노이즈 입력
        q_out = self.qnn(x)
        return self.fc(q_out)

if __name__ == "__main__":
    model = QuantumGenerator()
    print("\n[성공] 하이브리드 양자 생성자 모델이 구축되었습니다.")
    print(f"학습 가능한 양자 파라미터 수: {len(list(model.qnn.parameters()))}")
    
    # 테스트 입력 (잠재 노이즈)
    test_input = torch.rand(4)
    output = model(test_input)
    print(f"양자 생성자 출력 크기: {output.shape} (성공)")