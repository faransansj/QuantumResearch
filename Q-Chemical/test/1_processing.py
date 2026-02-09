import pandas as pd
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
