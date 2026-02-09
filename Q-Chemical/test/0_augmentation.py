from rdkit import Chem
import os

def augment_smiles(input_file, aug_factor=5):
    with open(input_file, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]
    augmented = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            new_set = {s}
            attempts = 0
            while len(new_set) < aug_factor and attempts < 50:
                new_set.add(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
                attempts += 1
            augmented.extend(list(new_set))
    with open('aug_data.txt', 'w') as f:
        for s in augmented:
            f.write(s + '\n')
    print(f"DONE: {len(augmented)} smiles saved to aug_data.txt")

if __name__ == "__main__":
    augment_smiles('oled_db.txt', aug_factor=5)
