import torch
import torch.nn as nn
from torchvision import models
import os

# 1. Setup Paths
BASE = r"C:\Users\Sanghamithra\Downloads\archive (2)\IESA_Hackathon_2026"
PTH_PATH = os.path.join(BASE, "models", "defect_model.pth")
ONNX_PATH = os.path.join(BASE, "models", "defect_model.onnx")

# 2. Re-create the Model Architecture
model = models.mobilenet_v3_small()

# --- NEW ROBUST ACCESS LOGIC ---
# We look into the classifier's first layer (index 0) to find the input features
if hasattr(model.classifier, 'in_features'):
    num_features = model.classifier.in_features
else:
    # If it's a Sequential block, get in_features from the first Linear layer
    num_features = model.classifier[0].in_features

print(f"Detected Input Features: {num_features}")

model.classifier = nn.Sequential(
    nn.Linear(num_features, 1024),
    nn.Hardswish(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, 8)
)
# -------------------------------

# 3. Load your trained weights
if not os.path.exists(PTH_PATH):
    print(f"❌ Error: Could not find {PTH_PATH}")
else:
    print("Loading trained weights...")
    model.load_state_dict(torch.load(PTH_PATH, map_location='cpu'))
    model.eval() 

    # 4. Create Dummy Input (Batch 1, 3 Channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 5. Export to ONNX
    print(f"Exporting to: {ONNX_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
    )

    print(f"✅ Final Success! ONNX model saved at: {ONNX_PATH}")
