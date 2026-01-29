import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchmetrics.classification import MultilabelAUROC
from PIL import Image
import pandas as pd
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class PercentCenterCrop:
    def __init__(self, keep):
        self.keep = keep

    def __call__(self, img):
        w, h = img.size
        new_w = int(w * self.keep)
        new_h = int(h * self.keep)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return img.crop((left, top, left + new_w, top + new_h))


class ChestXRayDataSet(Dataset):
    def __init__(self, images_dir, labels_csv, transform):
        self.transform = transform
        self.img_paths = []
        self.labels = []

        df = pd.read_csv(labels_csv)

        
        img_map = {}
        for folder in os.listdir(images_dir):
            folder_path = os.path.join(images_dir, folder)
            if folder.startswith("images_"):
                images_subdir = os.path.join(folder_path, "images")
                if os.path.isdir(images_subdir):
                    for img_name in os.listdir(images_subdir):
                        img_map[img_name] = os.path.join(images_subdir, img_name)

        for img_name in df["Image Index"]:
            self.img_paths.append(img_map[img_name])

        all_labels = df["Finding Labels"].str.split("|")
        self.classes = sorted(set(d for sub in all_labels for d in sub))
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}

        for diseases in all_labels:
            label = torch.zeros(len(self.classes))
            for d in diseases:
                label[self.class_to_index[d]] = 1
            self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]



transform = transforms.Compose([
    PercentCenterCrop(keep=1.0),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = ChestXRayDataSet(
    images_dir="E:/DataSets/CuraLink/archive (1)",
    labels_csv="E:/DataSets/CuraLink/archive (1)/Data_Entry_2017.csv",
    transform=transform
)

num_classes = len(dataset.classes)


class CheXNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.model(x)


model = CheXNet(num_classes).to(device)


model.load_state_dict(
    torch.load("best_chexnet_params.pth", map_location=device, weights_only=True)
)
model.eval()
print("Best model loaded")


criterion = nn.BCEWithLogitsLoss()
auroc = MultilabelAUROC(num_labels=num_classes, average=None).to(device)


VAL_SUBSET_SIZE = 5000   

indices = list(range(VAL_SUBSET_SIZE))
val_subset = Subset(dataset, indices)

val_loader = DataLoader(
    val_subset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


def evaluate(loader):
    auroc.reset()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).int()

            outputs = model(images)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            auroc.update(probs, labels)

    scores = auroc.compute()
    mean_auroc = scores.mean().item()
    avg_loss = running_loss / len(loader)

    print("Validation AUROC per class:")
    for cls, score in zip(dataset.classes, scores):
        print(f"{cls:<22}: {score:.4f}")

    print(f"Mean AUROC: {mean_auroc:.4f}")
    print(f"Val Loss  : {avg_loss:.4f}")

    return avg_loss, mean_auroc


if __name__ == "__main__":
    evaluate(val_loader)
