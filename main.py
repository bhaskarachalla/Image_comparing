import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove last layer

    def forward(self, x):
        return self.model(x)

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor()

    def forward(self, image1, image2):
        feature1 = self.feature_extractor(image1)
        feature2 = self.feature_extractor(image2)
        feature1 = feature1.view(feature1.size(0), -1)
        feature2 = feature2.view(feature2.size(0), -1)
        similarity = F.cosine_similarity(feature1, feature2)
        return similarity

# Instantiate the model
model = SiameseNetwork().to(device)
model.eval()

# Function to load and preprocess images
def load_image(image):
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to compute similarity score
def compute_similarity(image1, image2):
    with torch.no_grad():
        similarity_score = model(image1, image2)
    return similarity_score.item()

# Streamlit UI
st.title('Image Similarity Checker')

uploaded_image1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
uploaded_image2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

if uploaded_image1 is not None and uploaded_image2 is not None:
    image1 = load_image(Image.open(uploaded_image1))
    image2 = load_image(Image.open(uploaded_image2))
    
    if st.button("Check Similarity"):
        similarity_score = compute_similarity(image1, image2)
        st.write(f'Similarity Score: {similarity_score:.4f}')
