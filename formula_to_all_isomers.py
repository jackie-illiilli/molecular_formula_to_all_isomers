import molecules
import re
from rdkit import Chem

# 0 识别分子组成
def count_atoms(formula):
    """给定分子式的字符串，识别分子式中每一种元素的个数

    Args:
        formula (str): like "C2H5OH" or "C2H6O"

    Returns:
        dict: like {'C': 2, 'H': 6, 'O': 1}
    """    
    atoms = {}
    # 定义正则表达式，用于匹配元素符号和数量
    pattern = "([A-Z][a-z]*)(\d*)"
    for match in re.findall(pattern, formula):
        element, count = match
        if count == "":
            count = "1"
        count = int(count)
        atoms[element] = atoms.get(element, 0) + count
    # 返回结果字典
    return atoms

# 1 计算不饱和度
def calculate_unsaturation(molecule_formula):
    """
    计算有机化合物的不饱和度。

    参数：
    molecule_formula：dict，分子式中每个元素的数量。

    返回值：
    int，有机化合物的不饱和度。

    """

    # 定义一个字典，存储每个元素对应的原子价
    atom_value = {"C": 4, "N": 3, "O": 2, "H": 1}

    # 初始化 values 列表，其中 values[0] 存储非氢原子的数量，values[1] 存储应有的氢原子数量
    values = [0, 0]

    # 遍历分子式中的每个元素，计算 values[0] 和 values[1]
    for key in molecule_formula.keys():
        # 如果当前元素是氢，则将其数量加入 values[1]
        if atom_value[key] == 1:
            values[1] += molecule_formula[key]
        else:
            # 如果当前元素不是氢，则将其数量加入 values[0]，同时计算应有的氢原子数量并加入 values[1]
            values[0] += molecule_formula[key]
            values[1] += (4 - atom_value[key]) * molecule_formula[key]

    # 如果 values[1] 不是偶数，则无法计算不饱和度，抛出异常
    if values[1] % 2 != 0:
        raise ValueError("Invalid formula: the number of hydrogen atoms should be even.")

    # 计算不饱和度并返回结果
    unsaturation = values[0] - values[1] // 2 + 1
    return unsaturation

def configurational_isomers_finder(molecule_formula):
    # 计算分子组成的列表形式
    molecule_formula = count_atoms(molecule_formula)
    # 计算不饱和度
    unsaturation = calculate_unsaturation(molecule_formula)
    # 分析非氢原子的个数和种类
    all_heavy_atoms_num = sum([molecule_formula[key] for key in molecule_formula.keys() if key != "H"])
    all_heavy_atom = set(list(molecule_formula.keys()))
    assert all_heavy_atoms_num >= 1
    states = set(["C"])
    # 把所有重原子当成碳，生成可能的所有结构
    for _ in range(all_heavy_atoms_num - 1):
        new_state = set()
        for smiles in states:
            new_state.update(molecules.get_valid_actions(smiles, {"C"}, allow_atom_addition=1, allow_bond_addition=0, allow_removal=0, allow_no_modification=0,
                        allowed_ring_sizes=[0], allow_bonds_between_rings=0, allow_atom_replace=False, allow_atom_replace_other=True, change_bond_Stereo=True, 
                        atom_bond_order={1 : Chem.BondType.SINGLE,}))
        states = new_state
    # 用杂原子替换碳
    for key in molecule_formula.keys():
        if key in ["H", "C"]:
            continue
        for _ in range(molecule_formula[key]):
            new_state = set()
            for smiles in states:
                new_state.update(molecules.get_valid_actions(smiles, all_heavy_atom, allow_atom_addition=0, allow_bond_addition=0, allow_removal=0, allow_no_modification=0,
                        allowed_ring_sizes=[0], allow_bonds_between_rings=0, allow_atom_replace=1, allow_atom_replace_other=0, change_bond_Stereo=True, 
                        atom_bond_order={1 : Chem.BondType.SINGLE,}, remove_new_atom={key}))
            states = new_state
    # 根据不饱和度补充环系结构和多重键     
    for _ in range(unsaturation):
        new_state = set()
        for smiles in states:
            new_state.update(molecules.get_valid_actions(smiles, all_heavy_atom, allow_atom_addition=0, allow_bond_addition=1, allow_removal=0, allow_no_modification=0,
                                allowed_ring_sizes=None, allow_bonds_between_rings=0, allow_atom_replace=0, allow_atom_replace_other=0, change_bond_Stereo=True, 
                                atom_bond_order={1 : Chem.BondType.SINGLE,}, remove_new_atom={key}, allow_valance=[1]))
        states = new_state
    return states

if __name__ == "__main__":
    all_smiles = configurational_isomers_finder("C6H6")
    all_smiles = list(set([Chem.MolToSmiles(Chem.MolFromSmiles(each)) for each in all_smiles]))
    print(len(all_smiles))
    # all_mols = [Chem.MolFromSmiles(each) for each in all_smiles]
    # Chem.Draw.MolsToGridImage(all_mols)