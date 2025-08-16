import torch
import esm
import os
import csv
import argparse
from bconformer.model import Conformer

def read_fasta(fasta_path):
    """
    Read sequences from a FASTA file.
    Returns a dictionary {sequence_id: sequence}
    """
    sequences = {}
    with open(fasta_path, "r") as f:
        seq_id, seq = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    sequences[seq_id] = "".join(seq)
                seq_id = line[1:].split()[0]  # take the first token after ">"
                seq = []
            else:
                seq.append(line)
        if seq_id is not None:
            sequences[seq_id] = "".join(seq)
    return sequences


def predict_epitopes(seq_id, sequence, model, esm_model, esm_alphabet, device, threshold=0.25, out_dir="predictions"):
    """
    Predict per-residue epitope scores for one sequence and save results to CSV.
    """
    os.makedirs(out_dir, exist_ok=True)

    # === Step 1: ESM embedding ===
    batch_converter = esm_alphabet.get_batch_converter()
    data = [(seq_id, sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        esm_outputs = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = esm_outputs["representations"][33]  # [1, L+2, 1280]

    embeddings = token_representations[:, 1:-1, :]  # remove CLS and EOS tokens
    embeddings = embeddings.transpose(1, 2)  # [1, 1280, L]

    # === Step 2: Conformer forward pass ===
    with torch.no_grad():
        logits = model(embeddings)  # [1, 2, L]
        probs = torch.softmax(logits, dim=1)[0, 1, :]  # probability of epitope class

    scores = probs.cpu().numpy()

    # === Step 3: Save results to CSV ===
    csv_path = os.path.join(out_dir, f"{seq_id}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Residue", "Score", "Predicted Label"])
        for i, (res, score) in enumerate(zip(sequence, scores), start=1):
            label = 1 if score >= threshold else 0
            writer.writerow([i, res, score, label])

    print(f"[OK] {seq_id} → {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict per-residue epitope scores from a FASTA file.")
    parser.add_argument("--fasta", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--out_dir", type=str, default="predictions", help="Output directory for CSV files.")
    parser.add_argument("--threshold", type=float, default=0.25, help="Threshold for epitope prediction (default: 0.25).")
    parser.add_argument("--model_path", type=str, default="src/model/bconformer_1.pth", help="Path to trained Conformer model.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load pretrained ESM model ===
    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()

    # === Load trained Conformer model ===
    model = Conformer(in_chans=1280, num_classes=2)
    state = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # === Read FASTA and run predictions ===
    sequences = read_fasta(args.fasta)
    for seq_id, seq in sequences.items():
        predict_epitopes(seq_id, seq, model, esm_model, esm_alphabet, device, args.threshold, args.out_dir)


if __name__ == "__main__":
    main()
