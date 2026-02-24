import os
import os.path as osp
import hashlib
from typing import List, Tuple, Union

import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import to_dense_adj
from .utils import load_pickle, save_pickle, node_counts
from .mol_utils import mol_to_torch_geometric, mol2smiles
from evaluation.moses.molecules import build_molecule
from loguru import logger

TRAIN_HASH = '05ad85d871958a05c02ab51a4fde8530'
VALID_HASH = 'e53db4bff7dc4784123ae6df72e3b1f0'
TEST_HASH = '677b757ccec4809febd83850b43e1616'


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def compare_hash(output_file: str, correct_hash: str) -> bool:
    """
    Computes the md5 hash of a SMILES file and check it against a given one
    Returns false if hashes are different
    """
    output_hash = hashlib.md5(open(output_file, 'rb').read()).hexdigest()
    if output_hash != correct_hash:
        print(f'{output_file} file has different hash, {output_hash}, than expected, {correct_hash}!')
        return False

    return True


atom_encoder = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
    "B": 4,
    "Br": 5,
    "Cl": 6,
    "I": 7,
    "P": 8,
    "S": 9,
    "Se": 10,
    "Si": 11,
}
atom_decoder = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]


class GuacamolDataset(InMemoryDataset):
    train_url = 'https://figshare.com/ndownloader/files/13612760'
    test_url = 'https://figshare.com/ndownloader/files/13612757'
    val_url = 'https://figshare.com/ndownloader/files/13612766'
    all_url = 'https://figshare.com/ndownloader/files/13612745'

    def __init__(self, root, split, filter_dataset=True, transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.filter_dataset = filter_dataset
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        self.smiles = load_pickle(osp.join(self.processed_dir, f'{split}_smiles.pickle'))
        self.num_nodes = load_pickle(osp.join(self.processed_dir, 'node_counts.pickle'))[split]
        self.node_types = torch.tensor([7.4090e-01, 1.0693e-01, 1.1220e-01, 1.4213e-02, 6.0579e-05, 1.7171e-03,
        8.4113e-03, 2.2902e-04, 5.6947e-04, 1.4673e-02, 4.1532e-05, 5.3416e-05])
        self.edge_types = torch.tensor([9.2526e-01, 3.6241e-02, 4.8489e-03, 1.6513e-04, 3.3489e-02])

    @property
    def raw_file_names(self):
        return ['train.smiles', 'val.smiles', 'test.smiles']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        import rdkit  # noqa
        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, 'train.smiles'))

        valid_path = download_url(self.val_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, 'val.smiles'))

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, 'test.smiles'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        splits = ['train', 'val', 'test']
        counts = {split: 0 for split in splits}
        for split in splits:
            smile_list = open(osp.join(self.raw_dir, f'{split}.smiles')).readlines()

            pbar = tqdm(total=len(smile_list))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            smiles_kept = []
            for i, smile in enumerate(tqdm(smile_list)):
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    data = mol_to_torch_geometric(mol, atom_encoder, smile)
                    assert data.x.dim() == 2 and data.edge_index.dim() == 2 and data.edge_attr.dim() == 2, f'x : {data.x.dim()}, edge_index : {data.edge_index.dim()}, edge_attr : {data.edge_attr.dim()}'
                    if self.filter_dataset:
                        node_attr = data.x[..., 0]
                        edge_targets = to_dense_adj(
                            edge_index=data.edge_index,
                            edge_attr=data.edge_attr[..., 0] + 1,
                            max_num_nodes=node_attr.shape[0]
                        ).squeeze() if data.edge_index.numel() > 0 else torch.empty((2, 0)).long()
                        molecule = build_molecule(atom_types=node_attr, edge_types=edge_targets, atom_decoder=atom_decoder)
                        smiles = mol2smiles(molecule)
                        if smiles is not None:
                            try:
                                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                                if len(mol_frags) == 1:
                                    if self.pre_filter is not None and not self.pre_filter(data):
                                        continue
                                    if self.pre_transform is not None:
                                        data = self.pre_transform(data)
                                    data_list.append(data)
                                    smiles_kept.append(smile)
                            except Chem.rdchem.AtomValenceException:
                                print("Valence error in GetmolFrags")
                            except Chem.rdchem.KekulizeException:
                                print("Can't kekulize molecule")
                    else:
                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        data_list.append(data)
                        smiles_kept.append(smile)
                pbar.update(1)

            pbar.close()

            node_count = node_counts(data_list)
            counts[split] = node_count
            print(f"Number of smiles kept: {len(smiles_kept)} / {len(smile_list)}")
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
            save_pickle(set(smiles_kept), osp.join(self.processed_dir, f'{split}_smiles.pickle'))
        save_pickle(counts, osp.join(self.processed_dir, 'node_counts.pickle'))
