import torch
from torch import optim, nn
from c_classical_baseline import ClassicalBaseline

# 1. 고전 모델 및 판별자 설정
# 양자 생성자 대신 순수 고전 LSTM 생성자를 사용합니다.
generator = ClassicalBaseline()
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
    
    # 고전 생성자가 만든 가짜 데이터 판별
    latent_noise = torch.rand(4) 
    fake_data = generator(latent_noise)
    fake_output = discriminator(fake_data.detach())
    d_loss_fake = criterion(fake_output, torch.zeros_like(fake_output))
    
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()
    
    # --- 고전 생성자(Generator) 학습 ---
    g_optimizer.zero_grad()
    gen_output = discriminator(fake_data)
    g_loss = criterion(gen_output, torch.ones_like(gen_output))
    
    g_loss.backward()
    g_optimizer.step()
    
    return d_loss.item(), g_loss.item()

if __name__ == "__main__":
    print("\n--- [비교군] 순수 고전 LSTM GAN 학습 시작 ---")
    dummy_real_data = torch.rand(13) 
    
    for epoch in range(1, 11):
        d_loss, g_loss = train_step(dummy_real_data)
        if epoch % 2 == 0:
            print(f"Epoch [{epoch}/10] | Classical D_Loss: {d_loss:.4f} | Classical G_Loss: {g_loss:.4f}")

    print("\n[성공] 고전 Baseline 모델의 학습 로그가 확보되었습니다.")
    print("이제 이 데이터를 양자 하이브리드 모델의 결과와 비교할 수 있습니다.")