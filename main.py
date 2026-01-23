import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from model import DiabetesClassifier
import io

app = FastAPI()

device = torch.device("cpu")
MODEL_PATH = "models/D_model.pt"

# Load MobileNetV2 base model (NO pretrained)
base_model = models.mobilenet_v2(pretrained=False)

# Create classifier
model = DiabetesClassifier(base_model)

# Load weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Image preprocessing (MUST match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"status": "Diabetes Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    class_names = ["diabetes" , "non_diabetes"]
    return {"prediction": class_names[predicted.item()]}
