import esm
from esm import Alphabet
from embed import *
from model import *
from train_eval import *


# =================================
'''
dataset folders
'''

fasta_files = "..." # directory containing training (or evaluating) fastas
pdb_files = "..." # directory containing training (or evaluating) pdbs


# =================================
'''
ESM embedding for all antigen chains
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model = esm_model.to(device)
esm_model.eval()

dataset = EpitopeDataset(fasta_files, pdb_files, esm_model, esm_alphabet, device)

# print("Number of antigen chains:", EpitopeDataset.TOTAL_ANTIGEN_CHAINS)
# print("Number of antibody chains:", EpitopeDataset.TOTAL_ANTIBODY_CHAINS)


# =================================
'''
dataloader
'''

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_padding)

for i, batch in enumerate(dataloader):
    embedding = batch["embedding"]        # shape: [B, max_len, 1280]
    labels = batch["labels"]              # shape: [B, max_len]
    mask = batch["mask"]                  # shape: [B, max_len]
    attention_mask = batch["attention_mask"]  # shape: [B, max_len]

    # print(f"Embedding shape: {embedding.shape}")
    # print(f"Labels shape: {labels.shape}")
    # print(f"Mask shape: {mask.shape}")
    # print(f"Attention mask shape: {attention_mask.shape}")
    #
    #
    # print("\n=== A sequence sample ===")
    # print(f"Embedding shape: {embedding[0].shape}")  # [max_len, 1280]
    # print(f"Embedding:\n{embedding[0]}")
    # print(f"Labels:\n{labels[0]}")
    # print(f"Mask:\n{mask[0]}")

    break


# =================================
'''
Conformer
'''

model = Conformer(in_chans=1280, num_classes=2, depth=12)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# =================================
'''
training with checkpoints
'''

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()
threshold = 0.3
epochs = 150
all_metrics = []

save_dir = "..."  # directory saving models
os.makedirs(save_dir, exist_ok=True)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_stats = train_one_epoch(model, criterion, dataloader, optimizer, device, epoch, scaler)
    val_stats, _ = evaluate(dataloader, model, device, threshold)

    all_metrics.append(val_stats)

    print(f"Train loss: {train_stats['loss']:.4f}\n")

    if 50 <= epoch + 1 <= 150:
        save_path = os.path.join(save_dir, f"model_epoch{epoch + 1}_AgIoU{val_stats['AgIoU']:.4f}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'all_metrics': all_metrics,
        }, save_path)


# =================================
'''
Evaluation with existing models
'''

model_name = "..."
model_path = os.path.join("...", model_name)
threshold = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Conformer(in_chans=1280, num_classes=2)
state = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(state)
model.to(device)

with torch.no_grad():
    metrics, sample_ious = evaluate(dataloader, model, device, threshold)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")