import os
import torch
import numpy as np
from epitope import *
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser, is_aa, Polypeptide
import esm
from esm import Alphabet


three_to_one_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'SEC': 'U', 'PYL': 'O', 'ASX': 'B', 'GLX': 'Z', 'UNK': 'X'
}


def esm_embed_sequences(sequences, model, alphabet, device):
    embeddings = []
    for seq in sequences:
        batch = alphabet.get_batch_converter()([("protein", seq)])
        batch_labels, batch_strs, batch_tokens = batch
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_embeddings = results["representations"][33]
        # Remove BOS and EOS tokens
        seq_embedding = token_embeddings[0, 1:-1].cpu()
        embeddings.append(seq_embedding)
    return torch.cat(embeddings, dim=0)


class EpitopeDataset(Dataset):
    def __init__(self, fasta_dir, pdb_dir, esm_model, esm_alphabet, device):
        TOTAL_ANTIGEN_CHAINS = 0
        TOTAL_ANTIBODY_CHAINS = 0

        self.fasta_dir = fasta_dir
        self.pdb_dir = pdb_dir
        self.esm_model = esm_model
        self.esm_alphabet = esm_alphabet
        self.device = device

        self.fasta_files = sorted([f for f in os.listdir(fasta_dir) if f.endswith('.fasta')])
        self.pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])

        self.antigen_len_cache = {}

        total_ag = 0
        total_ab = 0
        for fasta_file in self.fasta_files:
            ag_chains, ab_chains = parse_chains(fasta_file)
            total_ag += len(ag_chains)
            total_ab += len(ab_chains)

        EpitopeDataset.TOTAL_ANTIGEN_CHAINS = total_ag
        EpitopeDataset.TOTAL_ANTIBODY_CHAINS = total_ab

    def __len__(self):
        return len(self.fasta_files)

    def __getitem__(self, idx):
        if idx in self.antigen_len_cache:
            antigen_length = self.antigen_len_cache[idx]
        else:
            antigen_length = None

        fasta_name = self.fasta_files[idx]
        fasta_id = os.path.splitext(fasta_name)[0]

        matched_pdb_file = None
        for f in self.pdb_files:
            if fasta_id in f:
                matched_pdb_file = os.path.join(self.pdb_dir, f)
                break

        if matched_pdb_file is None:
            raise ValueError(f"No matching pdb file found for {fasta_name}")

        antigen_chains, antibody_chains = parse_chains(fasta_name)

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", matched_pdb_file)
        model = structure[0]

        sorted_chain_ids = sorted([chain.id for chain in model])
        assert len(sorted_chain_ids) == len(antigen_chains) + len(antibody_chains)

        antigen_chain_ids = sorted_chain_ids[:len(antigen_chains)]
        antibody_chain_ids = sorted_chain_ids[len(antigen_chains):]

        antigen_chains_objs = [model[c] for c in antigen_chain_ids]
        antibody_chains_objs = [model[c] for c in antibody_chain_ids]

        # Antigen length
        if antigen_length is None:
            length = 0
            for chain in antigen_chains_objs:
                for residue in chain.get_residues():
                    if is_aa(residue):
                        length += 1
            self.antigen_len_cache[idx] = length
            antigen_length = length

        # Antigen sequence
        antigen_sequences = []
        for chain in antigen_chains_objs:
            seq = ""
            for residue in chain.get_residues():
                if is_aa(residue):
                    try:
                        resname = residue.get_resname()
                        aa = three_to_one_dict.get(resname, 'X')
                        seq += aa
                    except KeyError:
                        continue
            antigen_sequences.append(seq)

        embedding = esm_embed_sequences(
            antigen_sequences, self.esm_model, self.esm_alphabet, self.device
        )

        labels = get_epitope_labels(antigen_chains_objs, antibody_chains_objs)
        mask = torch.ones(labels.shape[0], dtype=torch.bool)

        return {
            'embedding': embedding,
            'labels': labels,
            'mask': mask,
            'antigen_length': antigen_length
        }


def collate_fn_padding(batch, max_seq_len = 1024):
    batch_embeddings = []
    batch_labels = []
    batch_masks = []
    attn_masks = []

    for item in batch:
        L = item['embedding'].shape[0]
        pad_len = max_seq_len - L
        if pad_len < 0:
            continue

        embedding = F.pad(item['embedding'], (0, 0, 0, pad_len), value=0)
        labels = F.pad(item['labels'], (0, pad_len), value=-100)
        mask = F.pad(item['mask'], (0, pad_len), value=0)
        attn_mask = torch.cat([torch.ones(L), torch.zeros(pad_len)])

        batch_embeddings.append(embedding)
        batch_labels.append(labels)
        batch_masks.append(mask)
        attn_masks.append(attn_mask)

    batch_embeddings = torch.stack(batch_embeddings)
    batch_labels = torch.stack(batch_labels)
    batch_masks = torch.stack(batch_masks)
    attn_masks = torch.stack(attn_masks)

    return {
        "embedding": batch_embeddings,
        "labels": batch_labels,
        "mask": batch_masks,
        "attention_mask": attn_masks
    }
