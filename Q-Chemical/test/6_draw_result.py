import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os

def draw():
    # 1. 성공한 데이터 파일 확인 (5번 코드가 저장한 파일명)
    input_file = 'final_success_results.csv'
    
    if not os.path.exists(input_file):
        print(f"파일이 없습니다: {input_file}")
        return

    # 2. 데이터 로드 및 유효한 분자 필터링
    df = pd.read_csv(input_file)
    valid_mols = []
    legends = []
    
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol:
            valid_mols.append(mol)
            legends.append(f"QED: {row['qed']:.3f}")
    
    if valid_mols:
        # 3. 분자 구조 그리기 (하얀색 배경 강제 설정)
        img = Draw.MolsToGridImage(
            valid_mols, 
            molsPerRow=3, 
            subImgSize=(400, 400),
            legends=legends,
            useSVG=False # PNG 포맷 사용
        )
        # 이미지를 저장
        img.save('research_result.png')
        print(f"\n[🎉 성공] {len(valid_mols)}개의 분자를 'research_result.png'로 그렸습니다!")
        print("이제 왼쪽 탐색기에서 파일을 다시 클릭해 보세요.")
    else:
        print("그릴 수 있는 유효한 분자가 데이터에 없습니다.")

if __name__ == "__main__":
    draw()