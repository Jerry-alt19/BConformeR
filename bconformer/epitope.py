import os
import torch
from Bio import SeqIO
from Bio.PDB import PDBParser, is_aa, Polypeptide
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities


def parse_chains(fasta_name):
    """
    This function parses a FASTA file and returns connected antigen and antibody chains
    FASTA files are in Alphafold2 Multimer version
    e.g. 1a2y_ag_C_ab_A_B.fasta
    """
    base = fasta_name.replace('.fasta', '')
    parts = base.split('_')
    ag_idx = parts.index('ag')
    ab_idx = parts.index('ab')
    antigen_chains = parts[ag_idx+1:ab_idx]
    antibody_chains = parts[ab_idx+1:]
    return antigen_chains, antibody_chains


def get_atoms(chains):
    return [atom for chain in chains for atom in unfold_entities(chain, 'A') if atom.element != 'H']


def get_epitope_labels(antigen_chain_objs, antibody_chain_objs):
    antibody_atoms = get_atoms(antibody_chain_objs)
    ns = NeighborSearch(antibody_atoms)
    epitope_residues = set()

    for chain in antigen_chain_objs:
        for res in chain.get_residues():
            if not is_aa(res):
                continue
            for atom in res:
                if ns.search(atom.coord, 4):
                    epitope_residues.add((chain.id, res.id))
                    break

    labels = []
    for chain in antigen_chain_objs:
        for res in chain.get_residues():
            if not is_aa(res):
                continue
            label_val = 1 if (chain.id, res.id) in epitope_residues else 0
            labels.append(label_val)
    return torch.tensor(labels, dtype=torch.long)


def classify_linear_epitopes(label_tensor):
    """
    Linear epitopes are extracted from linear regions;
    Regions are defined as linear stretches of antigen sequence having at least three antibody-contacting residues;
    Gaps between contacting residues are allowed, and a gap size of up to three non-contacting residues is chosen.

    Args:
        label_tensor (torch.Tensor or list[int]):
            A sequence of residue labels where:
              - 1 = contacting residue (epitope),
              - 0 = non-contacting residue,
              - -100 = padding / ignored.

    Returns:
        list[bool]:
            A boolean mask (same length as input) where `True` indicates
            residues that are part of a linear epitope, and `False` otherwise.
    """
    labels = label_tensor.tolist()
    length = len(labels)
    linear_flags = [False] * length
    visited = [False] * length

    i = 0
    while i < length:
        if labels[i] != 1 or visited[i]:
            i += 1
            continue

        segment_indices = []
        one_count = 0
        zero_count = 0
        j = i

        while j < length:
            if labels[j] == -100:
                break
            if labels[j] == 1:
                segment_indices.append(j)
                one_count += 1
                zero_count = 0
            elif labels[j] == 0:
                zero_count += 1
                if zero_count > 3:
                    break
                segment_indices.append(j)
            j += 1

        ones_in_segment = [idx for idx in segment_indices if labels[idx] == 1]
        if len(ones_in_segment) >= 3:
            for idx in ones_in_segment:
                linear_flags[idx] = True
                visited[idx] = True
            i = segment_indices[-1] + 1
        else:
            visited[i] = True
            i += 1

    return linear_flags