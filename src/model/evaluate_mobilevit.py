import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tqdm import tqdm 

# Config
BATCH_SIZE = 16
MODEL_PATH = "/home/nmichelotti/Desktop/Nanu Project/src/model/mobilevit_model.pth"
DATA_PATH = "/home/nmichelotti/Desktop/Nanu Project/data/model_dataset/test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms & Dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Load Model
model = create_model("mobilevit_s", pretrained=False)
model.reset_classifier(num_classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Evaluate
all_probs = []
all_labels = []

with torch.no_grad():
    for x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(DEVICE)
        logits = model(x).squeeze().cpu()
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.tolist())
        all_labels.extend(y.tolist())

# Metrics
binary_preds = [1 if p >= 0.5 else 0 for p in all_probs]

auc = roc_auc_score(all_labels, all_probs)
acc = accuracy_score(all_labels, binary_preds)
tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print Results
print("\n📊 Model Evaluation")
print(f"AUC:         {auc:.4f}")
print(f"Accuracy:    {acc:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
