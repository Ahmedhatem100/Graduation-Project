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
    # 1. READ & CONVERT
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file. Please upload a JPEG or PNG."}

    # 2. TRANSFORM
    # Ensure this image tensor shape becomes [1, 3, 224, 224]
    input_tensor = transform(image).unsqueeze(0)

    # 3. PREDICT WITH PROBABILITIES
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Apply Softmax to get percentages (0.0 to 1.0)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the top confidence and the class index
        confidence, predicted_idx = torch.max(probabilities, 1)

    # 4. DIAGNOSTIC OUTPUT
    class_names = ["diabetes", "non_diabetes"] # 0, 1
    
    return {
        "prediction": class_names[predicted_idx.item()],
        "confidence_score": f"{confidence.item() * 100:.2f}%",
        "raw_probabilities": {
            "diabetes": f"{probabilities[0][0].item():.4f}",
            "non_diabetes": f"{probabilities[0][1].item():.4f}"
        },
        "debug_note": "If both probs are near 0.50, your model is random guessing."
    }