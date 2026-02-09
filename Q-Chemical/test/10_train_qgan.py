import torch
from torch import optim, nn
from q_hybrid_qgan_core import QuantumGenerator

# 1. 모델 및 최적화 도구 설정
generator = QuantumGenerator()
# 판별자(Discriminator): 고전 신경망(간단한 MLP 구조)
discriminator = nn.Sequential(
    nn.Linear(13, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

g_optimizer = optim.Adam(generator.parameters(), lr=0.01)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.01)
criterion = nn.BCELoss()

def train_step(real_data):
    # --- 판별자(Discriminator) 학습 ---
    d_optimizer.zero_grad()
    
    # 진짜 데이터 판별
    real_output = discriminator(real_data)
    d_loss_real = criterion(real_output, torch.ones_like(real_output))
    
    # 양자 생성자가 만든 가짜 데이터 판별
    latent_noise = torch.rand(4) # 4차원 노이즈 입력
    fake_data = generator(latent_noise)
    fake_output = discriminator(fake_data.detach())
    d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output))
    
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()
    
    # --- 양자 생성자(Generator) 학습 ---
    # 판별자를 속이도록 양자 회로의 가중치(theta)를 업데이트합니다.
    g_optimizer.zero_grad()
    
    gen_output = discriminator(fake_data)
    g_loss = criterion(gen_output, torch.ones_like(gen_output))
    
    g_loss.backward()
    g_optimizer.step()
    
    return d_loss.item(), g_loss.item()

if __name__ == "__main__":
    print("\n--- [고도화] 하이브리드 QGAN 통합 학습 시작 ---")
    # 간단한 예시 데이터 (실제 데이터셋의 한 배치라고 가정)
    dummy_real_data = torch.rand(13) 
    
    for epoch in range(1, 11):
        d_loss, g_loss = train_step(dummy_real_data)
        if epoch % 2 == 0:
            print(f"Epoch [{epoch}/10] | D_Loss: {d_loss:.4f} | G_Loss: {g_loss:.4f}")

    print("\n[성공] 양자 생성자가 고전 판별자와 경쟁하며 학습을 완료했습니다.")
    print("이 과정에서 양자 회로의 '얽힘 가중치'가 최적화되었습니다.")