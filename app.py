import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import time

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="AI Animal Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .model-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    .metric-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 50%, #ff9ff3 100%);
        color: white;
        padding: 1rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shine 3s ease-in-out infinite;
    }
    
    @keyframes shine {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(180deg); }
    }
    
    .prediction-box h2 {
        margin: 0.5rem 0;
        font-size: 1.8rem;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-box h3 {
        margin: 0.3rem 0;
        font-size: 1.4rem;
        font-weight: 600;
    }
    
    .prediction-box p {
        margin: 0.3rem 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CNN MODEL DEFINITION
# =========================
class CNN(nn.Module):
    def __init__(self, input_number, class_number):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_number, out_channels=8, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.fcl = nn.Linear(64 * 15 * 15, class_number)
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcl(x)
        return x

# =========================
# LOAD MODEL FUNCTION
# =========================
@st.cache_resource
def load_model(model_name, num_classes, model_path):
    try:
        if model_name == "ResNet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "EfficientNet-B0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == "MobileNetV2":
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == "Custom CNN":
            model = CNN(input_number=1, class_number=num_classes)
        else:
            st.error("Unsupported model selected.")
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# =========================
# MODEL INFORMATION
# =========================
model_info = {
    "ResNet18": {
        "description": "A powerful model designed for deep learning",
        "accuracy": "92.3%",
        "speed": "Fast"
    },
    "EfficientNet-B0": {
        "description": "Efficient and accurate image classification",
        "accuracy": "94.1%",
        "speed": "Medium"
    },
    "MobileNetV2": {
        "description": "Optimized for mobile devices",
        "accuracy": "90.8%",
        "speed": "Very Fast"
    },
    "Custom CNN": {
        "description": "Custom designed convolutional network",
        "accuracy": "88.5%",
        "speed": "Fast"
    }
}

# =========================
# CONFIGURATION
# =========================
try:
    test_dir = r"C:\Users\melis\OneDrive\Masaüstü\My_Pacific_Technology_internship_project\PaTek_FIRST10DAY\dataset\test"
    test_dataset = datasets.ImageFolder(test_dir)
    class_names = test_dataset.classes
except:
    class_names = ['bird', 'cat', 'dog']  # Fixed syntax error - added comma

model_paths = {
    "ResNet18": r"C:\Users\melis\OneDrive\Masaüstü\My_Pacific_Technology_internship_project\PaTek_FIRST10DAY\resnet18_rgb_model_colab.pth",
    "EfficientNet-B0": r"C:\Users\melis\OneDrive\Masaüstü\My_Pacific_Technology_internship_project\PaTek_FIRST10DAY\efficientnet_rgb_model_colab.pth",
    "MobileNetV2": r"C:\Users\melis\OneDrive\Masaüstü\My_Pacific_Technology_internship_project\PaTek_FIRST10DAY\mobilenetv2_rgb_model_colab.pth",
    "Custom CNN": r"C:\Users\melis\OneDrive\Masaüstü\My_Pacific_Technology_internship_project\PaTek_FIRST10DAY\cnn_model_colab.pth",
}

# =========================
# IMAGE TRANSFORMS
# =========================
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

transform_gray = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# =========================
# MAIN UI
# =========================

# Header
st.markdown("""
<div class="main-header">
    <h1>🐾 AI-Powered Animal Classifier</h1>
    <p>Automatically recognize animal species with advanced deep learning models</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### ⚙️ Model Settings")

# Model selection
st.sidebar.markdown("### 🤖 Model Selection")
model_choice = st.sidebar.radio(
    "Select the model you want to use:",
    list(model_info.keys()),
    format_func=lambda x: f"{x}"
)

# Selected model information
if model_choice:
    info = model_info[model_choice]
    st.sidebar.markdown(f"""
    <div class="model-card">
        <h4>{model_choice}</h4>
        <p><strong>Description:</strong> {info['description']}</p>
        <p><strong>Accuracy:</strong> {info['accuracy']}</p>
        <p><strong>Speed:</strong> {info['speed']}</p>
    </div>
    """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([1, 1])
        
with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Upload an animal image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Show uploaded image
        image_rgb = Image.open(uploaded_file).convert("RGB")
        image_rgb.thumbnail((280, 280))  # Maksimum 280x280, orantıyı korur
        st.image(image_rgb)


with col2:
    st.subheader("🔮 Prediction Results")
    
    if uploaded_file is not None:
        if st.button("🚀 Start Prediction", type="primary", use_container_width=True):
            # Progress indicator
            with st.spinner("Processing..."):
                model = load_model(model_choice, len(class_names), model_paths[model_choice])
                
                if model:
                    # Prepare image
                    if model_choice == "Custom CNN":
                        image = image_rgb.convert("L")
                        img_tensor = transform_gray(image).unsqueeze(0)
                    else:
                        img_tensor = transform_rgb(image_rgb).unsqueeze(0)
                    
                    # Make prediction
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    img_tensor = img_tensor.to(device)
                    
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    inference_time = time.time() - start_time
                    
                    # Show main result
                    probs_np = probs.cpu().numpy()
                    top_pred_idx = np.argmax(probs_np)
                    top_confidence = probs_np[top_pred_idx] * 100

                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🎯 {class_names[top_pred_idx].title()}</h2>
                        <h3>Confidence: {top_confidence:.1f}%</h3>
                        <p>Processed in {inference_time:.3f}s</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---------- ALL PREDICTIONS TABLE IMMEDIATELY BELOW ----------
                    results_df = pd.DataFrame({
                        'Animal': class_names,
                        'Probability (%)': probs_np * 100
                    }).sort_values('Probability (%)', ascending=False)

                    st.markdown("**All Predictions:**")
                    st.dataframe(
                        results_df.round(1),
                        use_container_width=True,
                        hide_index=True
                    )

                else:
                    st.error("❌ Model loading failed. Please check model paths.")
    else:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Upload Required</h4>
            <p>Please upload an image to start classification.</p>
        </div>
        """, unsafe_allow_html=True)

