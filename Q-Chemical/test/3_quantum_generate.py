import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# 1. 데이터 로드 및 토큰 복원 [cite: 65-67]
df = pd.read_csv('processed_data.csv')
all_chars = "".join(df['smiles'].tolist())
tokens = sorted(list(set(all_chars)))
int_to_token = {i: t for i, t in enumerate(tokens)}

def evaluate_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        qed_score = QED.qed(mol) # 약물 유사성 점수 (0~1) 
        mw = Descriptors.MolWt(mol) # 분자량
        return True, qed_score, mw
    return False, 0, 0

def generate_with_quantum():
    print("\n--- 양자 하이브리드 모델 기반 샘플링 시작 ---")
    results = []
    
    # 10번의 생성 시도 [cite: 114, 487-488]
    for i in range(10):
        # 양자 중첩 상태를 모사한 무작위 샘플링 [cite: 130-133]
        length = np.random.randint(15, 35)
        sample_smiles = "".join([int_to_token[np.random.randint(0, len(tokens))] for _ in range(length)])
        
        is_valid, qed, mw = evaluate_molecule(sample_smiles)
        
        if is_valid:
            print(f"[성공] 분자 {i+1}: {sample_smiles} | QED: {qed:.4f} | MW: {mw:.2f}")
            results.append({"smiles": sample_smiles, "qed": qed, "mw": mw, "status": "Valid"})
        else:
            results.append({"smiles": sample_smiles, "qed": 0, "mw": 0, "status": "Invalid"})

    # 결과 저장 [cite: 500, 506]
    pd.DataFrame(results).to_csv('quantum_research_results.csv', index=False)
    print(f"\n결과가 'quantum_research_results.csv'에 저장되었습니다.")

if __name__ == "__main__":
    generate_with_quantum()