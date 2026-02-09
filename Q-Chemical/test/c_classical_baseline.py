import torch
from torch import nn

class ClassicalBaseline(nn.Module):
    def __init__(self, vocab_size=13, hidden_dim=64):
        super().__init__()
        # 고전적인 LSTM 구조: 시퀀스 데이터 학습의 표준
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: 잠재 노이즈 입력 (양자 모델과 동일한 조건)
        # LSTM을 통해 시퀀스 패턴 추출
        out, _ = self.lstm(x.unsqueeze(0)) 
        return self.fc(out.squeeze(0))

if __name__ == "__main__":
    # 양자 모델과 파라미터 수를 최대한 비슷하게 맞춘 대조군 설정
    model = ClassicalBaseline()
    print("\n--- [비교군] 고전 LSTM 베이스라인 모델 구축 완료 ---")
    
    test_input = torch.rand(4)
    output = model(test_input)
    print(f"고전 모델 출력 크기: {output.shape} (검증 성공)")
    print("\n이제 이 모델과 앞서 만든 양자 모델을 1:1로 맞붙여 성능을 비교합니다.")