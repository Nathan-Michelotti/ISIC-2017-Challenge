
import os
import subprocess
import re
from tqdm import tqdm

# Settings
start_epoch = 5
end_epoch = 15
script_path = "train_mobilevit.py"

# Regex patterns to extract metrics
patterns = {
    "auc": re.compile(r"AUC:\s+([0-9.]+)"),
    "accuracy": re.compile(r"Accuracy:\s+([0-9.]+)"),
    "sensitivity": re.compile(r"Sensitivity:\s+([0-9.]+)"),
    "specificity": re.compile(r"Specificity:\s+([0-9.]+)")
}

# Track best run
best = {"epoch": 0, "auc": 0.0, "accuracy": 0.0, "sensitivity": 0.0, "specificity": 0.0}

# Run Training Loop with tqdm progress bar
for epoch in tqdm(range(start_epoch, end_epoch + 1), desc="Optimizing Epochs"):
    print(f"\n--- Running training with {epoch} epochs ---")

    result = subprocess.run(
        ["python", script_path],
        capture_output=True,
        text=True,
        env={**dict(os.environ), "EPOCHS": str(epoch)}
    )

    output = result.stdout
    print(output)

    # Extract metrics
    try:
        auc = float(patterns["auc"].search(output).group(1))
        accuracy = float(patterns["accuracy"].search(output).group(1))
        sensitivity = float(patterns["sensitivity"].search(output).group(1))
        specificity = float(patterns["specificity"].search(output).group(1))

        if auc > best["auc"]:
            best.update({
                "epoch": epoch,
                "auc": auc,
                "accuracy": accuracy,
                "sensitivity": sensitivity,
                "specificity": specificity
            })

    except Exception as e:
        print(f"Failed to parse metrics for epoch {epoch}: {e}")

# Print Best Result
print("\n✅ Best Result")
print(f"Epochs:     {best['epoch']}")
print(f"AUC:        {best['auc']:.4f}")
print(f"Accuracy:   {best['accuracy']:.4f}")
print(f"Sensitivity: {best['sensitivity']:.4f}")
print(f"Specificity: {best['specificity']:.4f}")

