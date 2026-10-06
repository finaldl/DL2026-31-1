import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from datasets import load_from_disk
from torch.utils.data import DataLoader
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch):
    images = torch.stack([transform(item["image"].convert("RGB")) for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return images, labels

def extract_features(model, dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=2)
    all_feats, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model(images)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)

def get_resnet50_backbone():
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    return model.to(device)

def get_dinov2_backbone():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    return model.to(device)


if __name__ == "__main__":
    os.makedirs("./features", exist_ok=True)

    datasets_to_run = {
        "pets_trainval": load_from_disk("./data/pets_trainval"),
        "pets_test": load_from_disk("./data/pets_test"),
        "eurosat_train": load_from_disk("./data/eurosat_train"),
        "eurosat_test": load_from_disk("./data/eurosat_test"),
    }

    backbones = {
        "resnet50": get_resnet50_backbone(),
        "dinov2": get_dinov2_backbone(),
    }

    for bname, model in backbones.items():
        for dname, ds in datasets_to_run.items():
            print(f"Extracting {bname} on {dname}...")
            feats, labels = extract_features(model, ds)
            out_path = f"./features/{bname}_{dname}.pt"
            torch.save({"features": feats, "labels": labels}, out_path)
            print(f"Saved {out_path} | shape: {feats.shape}")

    print("XONG toàn bộ feature extraction.")