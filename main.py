from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import DiabetesClassifier
from torchvision import models

# ================= LOAD MODEL =================

# Rebuild MobileNet same as training
base_model = models.mobilenet_v2(weights=None)
model = DiabetesClassifier(base_model)

# Load weights
model.load_state_dict(torch.load("D_model.pt", map_location="cpu"))

model.eval()

# ================= FASTAPI =================
app = FastAPI()

# Image preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ================= API ROUTES =================

@app.get("/")
def home():
    return {"message": "Diabetes Tongue API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prob = output.item()

    label = "non_diabetes" if prob > 0.5 else "diabetes"


    return {
        "prediction": label,
        "probability": prob
    }
