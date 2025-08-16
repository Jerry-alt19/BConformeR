import torch
import esm
import os
import csv
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


if __name__ == "__main__":
    fasta_file = "example.fasta"   # input FASTA file
    out_dir = "predictions"         # output directory for CSV files
    threshold = 0.3         # change the threshold to get a good result 😎😎😎

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load pretrained ESM model ===
    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.to(device)
    esm_model.eval()

    # === Load trained Conformer ===
    model_name = "bconformer_1.pth"
    model_path = os.path.join("src/model", model_name)
    model = Conformer(in_chans=1280, num_classes=2)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # === Read FASTA and run predictions ===
    sequences = read_fasta(fasta_file)
    for seq_id, seq in sequences.items():
        predict_epitopes(seq_id, seq, model, esm_model, esm_alphabet, device, threshold, out_dir)