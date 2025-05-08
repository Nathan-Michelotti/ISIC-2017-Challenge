import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tqdm import tqdm

# Config
BATCH_SIZE = 16
EPOCHS = 7
LR = 1e-4
DATA_PATH = "/home/nmichelotti/Desktop/Nanu Project/data/model_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transforms & Loaders
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = datasets.ImageFolder(f"{DATA_PATH}/train", transform=transform)
test_dataset = datasets.ImageFolder(f"{DATA_PATH}/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# MobileViT Model
model = create_model("mobilevit_s", pretrained=True)
model.reset_classifier(num_classes=1)
model = model.to(DEVICE)


# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# Evaluation on Test Set
model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x).squeeze().cpu()
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.tolist())
        all_labels.extend(y.tolist())

# Convert probabilities to binary using 0.5 threshold
binary_preds = [1 if p >= 0.5 else 0 for p in all_probs]

# Metrics
auc = roc_auc_score(all_labels, all_probs)
acc = accuracy_score(all_labels, binary_preds)

# Confusion matrix: TN, FP, FN, TP
tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
sensitivity = tp / (tp + fn)  # True Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate

# Print metrics
print("\nFinal Evaluation Metrics (threshold = 0.5):")
print(f"AUC:         {auc:.4f}")
print(f"Accuracy:    {acc:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")  # How well it detects melanoma
print(f"Specificity: {specificity:.4f}")  # How well it avoids false melanoma

# Save Model
torch.save(model.state_dict(), "mobilevit_model.pth")
print("Saved model to mobilevit_model.pth")
