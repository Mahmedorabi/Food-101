from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
from model_resnet import CheckpointedResNet152, LabelEncoder, load_classes

# Initialize FastAPI app
app = FastAPI(title="Food Image Classifier API")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = load_classes('classes.txt')  # Ensure classes.txt is in the same directory
encoder = LabelEncoder(CLASSES)

# Load the model globally
model = CheckpointedResNet152(num_classes=101)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('resnet152_food21_best.pt', map_location=device))
model = model.to(device)
model.eval()

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the class name of an uploaded image using LabelEncoder.
    
    Args:
        file: Uploaded image file (jpg, jpeg, png)
    
    Returns:
        JSON response with predicted class name, confidence, and top 5 probabilities
    """
    try:
        # Read and process the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)[0]
            _, predicted_idx = torch.max(output, 1)
            predicted_idx = predicted_idx.item()
        
        confidence = probabilities[predicted_idx].item()
        predicted_class_name = encoder.get_label(predicted_idx)  # Map index to name
        if confidence < 0.5 :
            raise HTTPException(
                status_code=400,
                detail="This image is not food-related or the model is not confident enough"
                    )
        # Prepare response
        response = {
            "predicted_class_name": predicted_class_name,
            "confidence": confidence
        }
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

