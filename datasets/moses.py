import os
import os.path as osp
from typing import List, Tuple, Union

import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url
from .utils import load_pickle, save_pickle, node_counts
from .mol_utils import mol_to_torch_geometric


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


atom_encoder = {"C": 0, "N": 1, "S": 2, "O": 3, "F": 4, "Cl": 5, "Br": 6}
atom_decoder = ["C", "N", "S", "O", "F", "Cl", "Br"]


class MOSESDataset(InMemoryDataset):
    train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
    val_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'
    test_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv'

    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        self.smiles = load_pickle(osp.join(self.processed_dir, f'{split}_smiles.pickle'))
        self.num_nodes = load_pickle(osp.join(self.processed_dir, 'node_counts.pickle'))[split]
        self.node_types = torch.tensor([0.722338, 0.13661, 0.163655, 0.103549, 0.1421803, 0.005411, 0.00150])
        self.edge_types = torch.tensor([0.89740, 0.0472947, 0.062670, 0.0003524, 0.0486])

    @property
    def raw_file_names(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        import rdkit  # noqa
        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, osp.join(self.raw_dir, 'train.csv'))

        valid_path = download_url(self.val_url, self.raw_dir)
        os.rename(valid_path, osp.join(self.raw_dir, 'val.csv'))

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, osp.join(self.raw_dir, 'test.csv'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        splits = ['train', 'val', 'test']
        counts = {split: 0 for split in splits}
        for split in splits:
            smile_list = pd.read_csv(osp.join(self.raw_dir, f'{split}.csv'))['SMILES'].values

            pbar = tqdm(total=len(smile_list))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            smiles_kept = []
            for i, smile in enumerate(smile_list):
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    data = mol_to_torch_geometric(mol, atom_encoder, smile)
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
