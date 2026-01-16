from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI()

# --------------------------------------------------
# Device (CPU only – free & safe)
# --------------------------------------------------
device = torch.device("cpu")

# --------------------------------------------------
# Model Definition (MUST match training)
# --------------------------------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# --------------------------------------------------
# Load Trained Weights (state_dict)
# --------------------------------------------------
state_dict = torch.load("diabetes_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# --------------------------------------------------
# Class Names
# --------------------------------------------------
class_names = ["diabetes", "non_diabetes"]

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return {
        "prediction": class_names[pred.item()],
        "confidence": round(confidence.item() * 100, 2)
    }


@app.get("/")
def root():
    return {"status": "API running"}
