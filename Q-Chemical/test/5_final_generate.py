import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED

# 1. 학습 데이터 로드
df = pd.read_csv('processed_data.csv')
train_smiles = df['smiles'].tolist()
# 안전한 추가 토큰 후보
safe_tokens = ['C', 'O', 'N', '(C)', '(O)']

def smart_generate():
    print("\n--- 스마트 분자 생성 시작 (성공할 때까지 시도) ---")
    results = []
    attempts = 0
    success_count = 0
    
    # 목표: 유효한 분자 5개 찾기
    while success_count < 5 and attempts < 500:
        attempts += 1
        # 기존의 멀쩡한 분자 하나 선택
        base_s = train_smiles[np.random.randint(0, len(train_smiles))]
        # 끝에 안전한 토큰 하나 추가 (가장 안전한 변형)
        new_s = base_s + safe_tokens[np.random.randint(0, len(safe_tokens))]
        
        mol = Chem.MolFromSmiles(new_s)
        if mol:
            # 원본과 너무 똑같으면 제외 (최소한의 신규성 확보)
            if new_s not in train_smiles:
                qed = QED.qed(mol)
                print(f"[✅ 성공!] 시도 {attempts}회차 | QED: {qed:.4f} | SMILES: {new_s}")
                results.append({"smiles": new_s, "qed": qed, "status": "Valid"})
                success_count += 1
        
    if success_count == 0:
        print("\n[실패] 죄송합니다. 더 많은 학습 데이터가 필요합니다.")
        # 실패 시 시각화를 위해 원본 데이터라도 저장
        results.append({"smiles": train_smiles[0], "qed": QED.qed(Chem.MolFromSmiles(train_smiles[0])), "status": "Valid_Original"})

    # 최종 결과 저장
    pd.DataFrame(results).to_csv('final_success_results.csv', index=False)
    print(f"\n--- 최종 성공: {success_count}개 분자 확보 ---")
    print("결과가 'final_success_results.csv'에 저장되었습니다.")

if __name__ == "__main__":
    smart_generate()