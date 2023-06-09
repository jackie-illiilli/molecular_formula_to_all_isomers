# Generate all possible isomers from the molecular formula

# 给定分子式，给出所有包含所有同分异构体（包含旋光异构体）的SMILES结构

Generate SMILES for all isomers. Input formula likes "C6H6" or "C2H6O" or "C2H5OH". Output likes ["c1ccccc1", "C=C=C1[C@H]2C[C@@H]12", "C1C2[C@@H]3[C@H]4[C@@H]2[C@@]143"...]

molecules.py from MolDQN: [Optimization of Molecules via Deep Reinforcement Learning | Scientific Reports (nature.com)](https://www.nature.com/articles/s41598-019-47148-x)

## Requirements

rdkit

## How to use

```
import formula_to_all_isomers
all_smiles = formula_to_all_isomers.configurational_isomers_finder("C6H12ClF")
all_smiles = list(set([Chem.MolToSmiles(Chem.MolFromSmiles(each)) for each in all_smiles]))
print(len(all_smiles))
all_mols = [Chem.MolFromSmiles(each) for each in all_smiles]
Chem.Draw.MolsToGridImage(all_mols, useSVG=1)

```

208
![1678618412636](image/README/output.svg)
