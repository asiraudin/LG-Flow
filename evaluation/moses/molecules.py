import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import re
from collections import Counter
from loguru import logger


allowed_bonds = {'H': {0: 1, 1: 0, -1: 0},
                 'C': {0: [3, 4], 1: 3, -1: 3},
                 'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},    # In QM9, N+ seems to be present in the form NH+ and NH2+
                 'O': {0: 2, 1: 3, -1: 1},
                 'F': {0: 1, -1: 0},
                 'B': 3, 'Al': 3, 'Si': 4,
                 'P': {0: [3, 5], 1: 4},
                 'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
                 'Cl': 1, 'As': 3,
                 'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class Molecule:
    def __init__(self, atom_types, bond_types, atom_decoder):
        """ atom_types: n      LongTensor
            charges: n         LongTensor
            bond_types: n x n  LongTensor
            atom_decoder: extracted from dataset_infos. """
        assert atom_types.dim() == 1 and atom_types.dtype == torch.long, f"shape of atoms {atom_types.shape} " \
                                                                         f"and dtype {atom_types.dtype}"
        assert bond_types.dim() == 2 and bond_types.dtype == torch.long, f"shape of bonds {bond_types.shape} --" \
                                                                         f" {bond_types.dtype}"
        assert len(atom_types.shape) == 1
        assert len(bond_types.shape) == 2

        self.atom_types = atom_types.long()
        self.bond_types = bond_types.long()

        self.rdkit_mol = build_molecule(self.atom_types, self.bond_types, atom_decoder)
        self.num_nodes = len(atom_types)
        self.num_atom_types = len(atom_decoder)


def build_molecule(atom_types, edge_types, atom_decoder, verbose=False):
    if verbose:
        print("\nbuilding new molecule")

    mol = Chem.RWMol()
    for atom in atom_types:
        if atom == -1:
            continue
        a = Chem.Atom(atom_decoder[int(atom.detach())])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.detach(), atom_decoder[atom.detach()])

    edge_types = torch.triu(edge_types)
    edge_types[edge_types == -1] = 0
    all_bonds = torch.nonzero(edge_types)
    for i, bond in enumerate(all_bonds):
        if int(bond[0].detach()) != int(bond[1].detach()):
            mol.AddBond(
                int(bond[0].detach()),
                int(bond[1].detach()),
                bond_dict[edge_types[int(bond[0]), int(bond[1])].detach()]
            )
            if verbose:
                print("bond added:",
                      int(bond[0].detach()),
                      int(bond[1].detach()),
                      edge_types[int(bond[0]), int(bond[1])].detach(),
                      bond_dict[edge_types[int(bond[0]), int(bond[1])].detach()])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if verbose:
                    print("atomic num of atom with a large valence", an)
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    # print("Formal charge added")
    return mol


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def graph_to_smiles(e_hat, x_hat, num_nodes, atom_decoder):
    bs = e_hat.shape[0]
    e_hat, x_hat = e_hat.argmax(dim=-1), x_hat.argmax(dim=-1)
    all_smiles = []
    error_message = Counter()
    for i in range(bs):
        n = num_nodes[i]
        e_i = e_hat[i, :n, :n]
        x_i = x_hat[i, :n]
        molecule = Molecule(atom_types=x_i, bond_types=e_i, atom_decoder=atom_decoder)
        rdmol = molecule.rdkit_mol

        try:
            mol_frags = Chem.rdmolops.GetMolFrags(
                rdmol, asMols=True, sanitizeFrags=True
            )
            if len(mol_frags) > 1:
                error_message[4] += 1
            largest_mol = max(
                mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms()
            )
            Chem.SanitizeMol(largest_mol)
            smiles = Chem.MolToSmiles(largest_mol)
            all_smiles.append(smiles)
            error_message[-1] += 1
        except Chem.rdchem.AtomValenceException:
            error_message[1] += 1
            all_smiles.append('error')
        except Chem.rdchem.KekulizeException:
            error_message[2] += 1
            all_smiles.append('error')
        except Chem.rdchem.AtomKekulizeException or ValueError:
            error_message[3] += 1
            all_smiles.append('error')
    logger.info(
        f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
        f" -- No error {error_message[-1]}"
    )

    return all_smiles
