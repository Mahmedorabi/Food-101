import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from model_resnet import *

# Load model, encoder, and transform (same as above)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckpointedResNet152(num_classes=101)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('resnet152_food21_best.pt', map_location=device))
model = model.to(device)
model.eval()

classes = load_classes('classes.txt')
encoder = LabelEncoder(classes)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.title("Food Image Classifier")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image',  use_container_width=True)
    
    image = transform(image.convert('RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]
        _, predicted_idx = torch.max(output, 1)
        predicted_idx = predicted_idx.item()
    
    predicted_class = encoder.get_label(predicted_idx)
    confidence = probabilities[predicted_idx].item()
    
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.4f}")