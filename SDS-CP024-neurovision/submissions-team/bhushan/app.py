import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Set page config
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# Load the trained model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 112 * 112, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load model weights
model = SimpleCNN()
try:
    model.load_state_dict(torch.load("best_model_params.pt", map_location=torch.device("cpu")))
    model.eval()
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error Loading Model: {e}")

# Define preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Streamlit UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan to check for a brain tumor.")

# Upload Image
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("📸 **Image Uploaded Successfully!**")

    # Preprocess image
    input_tensor = preprocess_image(image)
    st.write(f"🔍 **Preprocessed Image Tensor Shape:** {input_tensor.shape}")  # Should be (1, 3, 224, 224)

    # Perform model inference
    with torch.no_grad():
        output = model(input_tensor)
        st.write(f"🧪 **Raw Model Output:** {output}")

    # Get prediction
    predicted_class = torch.argmax(output, dim=1).item()
    prediction_label = "Tumor Detected" if predicted_class == 1 else "No Tumor Detected"

    # Display prediction
    st.markdown(f"### 🏥 **Prediction:** `{prediction_label}`")

    # Confidence Score (Softmax)
    softmax = torch.nn.functional.softmax(output, dim=1)
    confidence = softmax[0, predicted_class].item() * 100
    st.write(f"🎯 **Confidence Score:** `{confidence:.2f}%`")

