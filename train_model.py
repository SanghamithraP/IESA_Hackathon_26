import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# Paths
BASE = r"C:\Users\Sanghamithra\Downloads\archive (2)\IESA_Hackathon_2026"
DATA_DIR = os.path.join(BASE, "data", "processed")
SAVE_PATH = os.path.join(BASE, "models", "defect_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Transforms
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Data Loaders
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Model Setup
print(f"🚀 Training starting on {DEVICE}...")
model = models.mobilenet_v3_small(weights="DEFAULT")

# Correct way to access in_features for MobileNetV3
# The classifier is a Sequential block; the last Linear layer is at index [3]
# But we usually take the input features from the first Linear layer at index [0]
num_features = model.classifier[0].in_features

# Rebuild the classifier for 8 classes
model.classifier = nn.Sequential(
    nn.Linear(num_features, 1024),
    nn.Hardswish(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, 8)
)

model = model.to(DEVICE)

# 4. Training Components
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 5. Training Loop (Reduced to 3 epochs for a quick first test)
num_epochs = 3 
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress every 100 batches
        if i % 100 == 0:
            print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} Summary | Loss: {running_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")

# 6. Save final model
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"🎉 Training Complete! Model saved as {SAVE_PATH}")
