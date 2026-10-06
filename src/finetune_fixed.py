"""
FIXED version of src/finetune.py (Phase 2 fine-tuning).

src/finetune.py is the ORIGINAL code that produced the 24 official results in
results/finetune_grid.csv. It is kept unchanged so the slide numbers stay
reproducible. This file contains the same pipeline with the problems found in
the audit fixed. Search for "# FIX #" to find every change.

  FIX #1  Stage 2 head bug: head params were a generator, used up by the Stage 1
          optimizer, so the Stage 2 head param group was EMPTY and the head never
          updated in Stage 2. Now a list -> head trains in both stages.
  FIX #2  Use the shared splits/pets_seed{s}_k{k}.json files (same images as the
          Phase 1 frozen grid). The original re-sampled at runtime and got
          different images at k=5/10/25 (only ~3% overlap with the JSON files).
  FIX #3  `import json` (needed by FIX #2).
  FIX #4  Seed every random source (head init, shuffle order, augmentation), not
          just the image selection, so a rerun gives (nearly) the same numbers.
  FIX #5  Keep the frozen backbone in eval() mode during Stage 1, so ResNet
          BatchNorm running statistics do not drift while it is "frozen".
  FIX #6  Write to separate outputs (results/finetune_grid_fixed.csv,
          runs_fixed/) so the official results are never overwritten/appended.
  FIX #7  Small: makedirs crash when the CSV path has no folder; comment that
          called our 20% split the "official test set".

The recipe (lr, weight decay, epochs, batch size, augmentation) is NOT changed:
these are bug fixes, not tuning. Results from this file are post-freeze and
are NOT the numbers on the slides.
"""
import os
import sys
import io
import csv
import gc
import json      # FIX #3: was missing, json.load() below would crash with NameError
import yaml
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from datasets import load_from_disk
from torch.utils.tensorboard import SummaryWriter

# Fix Windows console encoding for UTF-8 prints
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add current directory to sys.path for local module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sampler import make_kshot_splits

# FIX #6: separate outputs so the official results/finetune_grid.csv and runs/ stay untouched
CSV_PATH = "results/finetune_grid_fixed.csv"
RUNS_DIR = "runs_fixed"


# --- 1. CONFIG LOADER ---
def load_recipe(recipe_path="config/finetune_recipe.yaml"):
    """Loads hyperparameter recipe from YAML config."""
    if os.path.exists(recipe_path):
        with open(recipe_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        print(f"[Warning] Recipe file '{recipe_path}' not found. Using default fallback hyperparameters.")
        return {
            "resnet50": {
                "batch_size": 32, "lr_head": 1e-3, "lr_backbone": 1e-4,
                "weight_decay": 0.01, "epochs_stage1": 10, "epochs_stage2": 30
            },
            "dinov2": {
                "batch_size": 32, "lr_head": 1e-3, "lr_backbone": 1e-5,
                "weight_decay": 0.05, "epochs_stage1": 10, "epochs_stage2": 30
            }
        }


# --- 2. CSV LOGGING HELPER ---
def append_result_to_csv(filepath, row_dict):
    """Appends fine-tuning metrics to CSV, writing headers if file does not exist."""
    # FIX #7: os.makedirs("") crashes when filepath has no folder part
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    file_exists = os.path.exists(filepath)
    fieldnames = ["backbone", "dataset", "method", "k", "seed", "accuracy"]

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
    print(f"[CSV Log] Successfully saved result to '{filepath}'")


# --- 3. DATASET WRAPPER & TRANSFORMATIONS ---
class PetsPyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- 4. MODEL ARCHITECTURES ---
class DINOv2Classifier(nn.Module):
    def __init__(self, backbone, num_classes=37):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def build_model(model_name, num_classes=37):
    # FIX #1: list(...) instead of the bare .parameters() generator.
    # A generator can only be read once: the Stage 1 optimizer read it, so the
    # Stage 2 optimizer got an empty head group and the head never trained in Stage 2.
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model, list(model.fc.parameters()), [p for n, p in model.named_parameters() if "fc" not in n]
    elif model_name == "dinov2":
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model = DINOv2Classifier(backbone, num_classes=num_classes)
        return model, list(model.head.parameters()), [p for n, p in model.named_parameters() if "head" not in n]
    else:
        raise ValueError(f"Unknown backbone: {model_name}")


# --- 5. VRAM CLEANUP UTILITY ---
def cleanup_vram():
    """Force garbage collection and free CUDA memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# --- 6. TWO-STAGE FINE-TUNING ENGINE ---
def train_two_stage(model_name, k, seed, recipe, csv_path=CSV_PATH, device="cuda"):
    print(f"\n==========================================")
    print(f" Executing Run: Backbone={model_name} | k={k} | Seed={seed}")
    print(f"==========================================")
    # FIX #4: the original only used `seed` to pick images; head init, shuffle
    # order and augmentation were unseeded, so reruns gave different numbers.
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Parse recipe parameters
    cfg = recipe[model_name]
    batch_size = cfg["batch_size"]
    lr_head = float(cfg["lr_head"])
    lr_backbone = float(cfg["lr_backbone"])
    wd = float(cfg["weight_decay"])
    epochs_stage1 = cfg["epochs_stage1"]
    epochs_stage2 = cfg["epochs_stage2"]

    # Load datasets
    trainval_hf = load_from_disk("./data/pets_trainval")
    test_hf = load_from_disk("./data/pets_test")

    # FIX #2: read the shared split file so fine-tuning trains on the SAME images
    # as the Phase 1 frozen grid. The original called make_kshot_splits() here,
    # which picks different images than splits/*.json at k=5/10/25.
    split_file = f"splits/pets_seed{seed}_k{k}.json"
    if os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        train_indices = split_data["train"] if isinstance(split_data, dict) and "train" in split_data else split_data
    else:
        print(f"[Warning] {split_file} not found; generating this split from the sampler.")
        splits = make_kshot_splits(trainval_hf, k_values=[k], seed=seed)
        train_indices = splits[k if k == "all" else int(k)]

    train_ds = Subset(PetsPyTorchDataset(trainval_hf, transform=train_transform), train_indices)
    test_ds = PetsPyTorchDataset(test_hf, transform=val_transform)

    # num_workers=0 for Windows stability, pin_memory for faster GPU transfers
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model, loss, and mixed precision scaler
    cleanup_vram()
    model, head_params, backbone_params = build_model(model_name)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device='cuda')
    writer = SummaryWriter(f"{RUNS_DIR}/{model_name}_k{k}_seed{seed}")  # FIX #6

    # STAGE 1: Freeze Backbone, Train Head Only
    print(f"--> Stage 1: Training FC Head ({epochs_stage1} Epochs)")
    for p in backbone_params:
        p.requires_grad = False
    opt1 = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=wd)

    for epoch in range(epochs_stage1):
        # FIX #5: requires_grad=False stops gradients, but model.train() still
        # updates BatchNorm running mean/var. Put the frozen backbone in eval()
        # so it is truly frozen; only the head is in train mode.
        if model_name == "resnet50":
            model.eval()
            model.fc.train()
        else:
            model.backbone.eval()
            model.head.train()
        total_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt1.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                loss = criterion(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.step(opt1)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))
        writer.add_scalar("Loss/Stage1", avg_loss, epoch)
        print(f"    Epoch {epoch+1}/{epochs_stage1} - Loss: {avg_loss:.4f}")

    # STAGE 2: Unfreeze Backbone, Full Fine-Tuning
    print(f"\n--> Stage 2: Full Fine-Tuning ({epochs_stage2} Epochs)")
    for p in backbone_params:
        p.requires_grad = True
    opt2 = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head}  # now non-empty thanks to FIX #1
    ], weight_decay=wd)

    for epoch in range(epochs_stage2):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt2.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                loss = criterion(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.step(opt2)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))
        writer.add_scalar("Loss/Stage2", avg_loss, epoch)
        if (epoch + 1) % 5 == 0 or epoch == epochs_stage2 - 1:
            print(f"    Epoch {epoch+1}/{epochs_stage2} - Loss: {avg_loss:.4f}")

    # FIX #7: this is our 20% held-out split of the HF train set, not the official Pets test set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            with torch.amp.autocast(device_type='cuda'):
                preds = model(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

    acc = (correct / total) * 100.0
    print(f"\n--> Completed Run | Top-1 Test Accuracy: {acc:.2f}%")
    writer.close()

    # Append metrics to CSV
    append_result_to_csv(csv_path, {
        "backbone": model_name,
        "dataset": "pets",
        "method": "finetune",
        "k": str(k),
        "seed": seed,
        "accuracy": f"{acc:.2f}"
    })

    # Cleanup VRAM after run (critical between grid runs)
    del model, criterion, scaler, opt1, opt2
    cleanup_vram()

    return acc


# --- 7. FULL 24-RUN GRID EXECUTION ---
def run_full_grid(recipe):
    backbones = ["resnet50", "dinov2"]
    k_values = [5, 10, 25, "all"]
    seeds = [0, 1, 2]

    total_runs = len(backbones) * len(k_values) * len(seeds)
    current_run = 0

    print(f"\n==========================================")
    print(f" STARTING FULL FINE-TUNING GRID ({total_runs} RUNS)")
    print(f"==========================================")

    for model_name in backbones:
        for k in k_values:
            for seed in seeds:
                current_run += 1
                print(f"\nProgress: [{current_run}/{total_runs}]")
                train_two_stage(model_name, k=k, seed=seed, recipe=recipe)


# --- 8. CLI ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zam Phase 2 Fine-Tuning Pipeline (fixed)")
    parser.add_argument("--grid", action="store_true", help="Run the complete 24-run fine-tuning matrix")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "dinov2"])
    parser.add_argument("--k", type=str, default="10", help="Label budget: 5, 10, 25, or all")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0, 1, 2)")

    args = parser.parse_args()
    recipe = load_recipe()

    if not torch.cuda.is_available():
        raise SystemError("CUDA GPU is required for Phase 2 fine-tuning.")

    if args.grid:
        run_full_grid(recipe)
    else:
        k_val = "all" if args.k == "all" else int(args.k)
        train_two_stage(args.backbone, k=k_val, seed=args.seed, recipe=recipe)
