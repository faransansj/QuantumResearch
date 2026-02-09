import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap

def build_zz_feature_map():
    print("\n--- [고도화] ZZFeatureMap 기반 양자 얽힘 인코딩 설계 ---")
    
    # 4개의 토큰 특징을 입력받는 4-큐비트 시스템 설계
    # entanglement='full'은 모든 큐비트를 서로 복잡하게 얽히게 만듭니다.
    feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement='full')
    
    # 논문용 수치 데이터 추출
    print(f"회로의 깊이(Depth): {feature_map.depth()}")
    print(f"사용된 논리 게이트 수: {feature_map.count_ops()}")
    
    # 회로 구조 시각화 (텍스트 모드)
    print("\n[양자 회로 구조: 데이터 간의 얽힘 확인]")
    print(feature_map.decompose().draw(output='text'))
    
    print("\n이 구조는 단순 각도 입력(RY)을 넘어, 데이터 간의 상관관계를")
    print("양자 얽힘 상태로 변환하는 'Quantum-Native' 인코딩 방식입니다.")

if __name__ == "__main__":
    build_zz_feature_map()