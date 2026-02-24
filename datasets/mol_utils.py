import pickle
import torch
from torch_geometric.data import Data
from rdkit import Chem

# allowable multiple choice node and edge features
allowable_features = {
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_is_conjugated_list': [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
        safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
        safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
        safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
        allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRingSize(3)),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRingSize(4)),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRingSize(5)),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRingSize(6)),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRingSize(7)),
        ]
    return atom_feature


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
        allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
        allowable_features['possible_is_in_ring_list'].index(bond.IsInRingSize(3)),
        allowable_features['possible_is_in_ring_list'].index(bond.IsInRingSize(4)),
        allowable_features['possible_is_in_ring_list'].index(bond.IsInRingSize(5)),
        allowable_features['possible_is_in_ring_list'].index(bond.IsInRingSize(6)),
        allowable_features['possible_is_in_ring_list'].index(bond.IsInRingSize(7)),
        ]
    return bond_feature


def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_is_conjugated_list'],
        allowable_features['possible_is_in_ring_list']
        ]))


def atom_feature_vector_to_dict(atom_feature):
    [degree_idx,
    formal_charge_idx,
    num_h_idx,
    hybridization_idx,
    is_aromatic_idx,
    is_in_ring_idx] = atom_feature

    feature_dict = {
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_features['possible_numH_list'][num_h_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }

    return feature_dict


def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx,
    is_conjugated_idx,
    is_in_ring_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
        'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }

    return feature_dict


def mol_to_torch_geometric(mol, atom_encoder, smiles):
    # adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    # edge_index = adj.nonzero().contiguous().T
    # bond_types = adj[edge_index[0], edge_index[1]]
    # bond_types[bond_types == 1.5] = 4
    # edge_attr = bond_types.long()

    # bonds
    num_bond_features = 7  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.Tensor(edges_list).long().T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.Tensor(edge_features_list).long()

    else:   # mol has no bonds
        edge_index = torch.empty((2, 0)).long()
        edge_attr = torch.empty((0, num_bond_features)).long()

    x = []
    for atom in mol.GetAtoms():
        atom_features = [atom_encoder[atom.GetSymbol()]]
        atom_features.extend(atom_to_feature_vector(atom))
        x.append(atom_features)
    x = torch.Tensor(x).long()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                smiles=smiles)
    return data


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

