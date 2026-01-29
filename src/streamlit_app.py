"""
🤟 BISINDO Alphabet Detection
Modern & Interactive Streamlit Application
"""

import streamlit as st
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import DETR
from utils.setup import get_classes, get_colors
from utils.boxes import rescale_bboxes
import numpy as np
import time
from streamlit_lottie import st_lottie
import requests
import json

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Deteksi Alfabet BISINDO",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS Styling - ELEGANT MIDNIGHT BLUE THEME
# -------------------------
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;600&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background with Elegant Gradient */
    .stApp {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        background-attachment: fixed;
    }
    
    /* Glassmorphism Container */
    .main-container {
        background: rgba(44, 83, 100, 0.15);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(113, 178, 128, 0.2);
        box-shadow: 0 8px 32px 0 rgba(15, 32, 39, 0.5);
        margin: 1rem 0;
    }
    
    /* Hero Title */
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 600;
        background: linear-gradient(120deg, #4CC9F0 0%, #71B280 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    /* Subtitle Quote */
    .hero-quote {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #E0E0E0;
        text-align: center;
        font-weight: 300;
        font-style: italic;
        margin-bottom: 2rem;
        opacity: 0.85;
        letter-spacing: 0.5px;
    }
    
    /* Custom Upload Box */
    .upload-section {
        background: rgba(76, 201, 240, 0.08);
        border: 2px dashed rgba(76, 201, 240, 0.4);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        background: rgba(76, 201, 240, 0.15);
        border-color: rgba(76, 201, 240, 0.7);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(76, 201, 240, 0.2);
    }
    
    /* Result Container */
    .result-container {
        background: rgba(44, 83, 100, 0.25);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(113, 178, 128, 0.3);
    }
    
    /* Prediction Badge */
    .prediction-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4CC9F0 0%, #71B280 100%);
        color: #0F2027;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(76, 201, 240, 0.4);
    }
    
    /* Info Cards */
    .info-card {
        background: rgba(44, 83, 100, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CC9F0;
        color: #E8E8E8;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        background: rgba(44, 83, 100, 0.3);
        transform: translateX(5px);
    }
    
    .info-card h3 {
        margin-top: 0;
        color: #4CC9F0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #4CC9F0 0%, #71B280 100%);
        color: #0F2027;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 201, 240, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 201, 240, 0.5);
        background: linear-gradient(135deg, #71B280 0%, #4CC9F0 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 32, 39, 0.95) 0%, rgba(44, 83, 100, 0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    /* Metrics */
    .metric-container {
        background: rgba(44, 83, 100, 0.25);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(76, 201, 240, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        border-color: rgba(76, 201, 240, 0.5);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #4CC9F0 0%, #71B280 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(232, 232, 232, 0.8);
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(44, 83, 100, 0.2);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #E0E0E0;
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4CC9F0 0%, #71B280 100%);
        color: #0F2027;
        font-weight: 600;
    }
    
    /* Floating Animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Smooth Glow Effect */
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 10px rgba(76, 201, 240, 0.3); }
        50% { box-shadow: 0 0 20px rgba(76, 201, 240, 0.6); }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 32, 39, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #4CC9F0 0%, #71B280 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #71B280 0%, #4CC9F0 100%);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Input styling */
    .stTextInput>div>div>input {
        background: rgba(44, 83, 100, 0.2);
        color: #E8E8E8;
        border: 1px solid rgba(76, 201, 240, 0.3);
        border-radius: 8px;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(44, 83, 100, 0.15);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Camera input styling */
    [data-testid="stCameraInput"] {
        border-radius: 12px;
        overflow: hidden;
    }
    
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# Lottie Animation Loader
# -------------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def load_lottie_local(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

# -------------------------
# Model Setup
# -------------------------
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DETR(num_classes=26)
    model = model.to(device)
    model.eval()
    model.load_pretrained('model/300_model.pt', map_location=device)
    return model, device

@st.cache_data
def get_model_info():
    CLASSES = get_classes()
    COLORS = get_colors()
    return CLASSES, COLORS

# -------------------------
# Image Processing Function
# -------------------------
def process_prediction(frame, model, device, CLASSES, COLORS):
    """
    Process image and return detection results
    """
    height, width, _ = frame.shape
    
    # Transforms
    transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Preprocess
    transformed = transforms(image=frame)
    img_tensor = torch.unsqueeze(transformed['image'], dim=0).to(device)
    
    # Inference
    with torch.no_grad():
        start_time = time.time()
        result = model(img_tensor)
        inference_time = (time.time() - start_time) * 1000  # ms
    
    # Post-processing
    probabilities = result['pred_logits'].softmax(-1)[0, :, :-1].cpu()
    max_probs, max_classes = probabilities.max(-1)
    top_score, top_idx = max_probs.max(0)
    
    detection_info = {
        'detected': False,
        'class': None,
        'confidence': 0.0,
        'inference_time': inference_time,
        'frame': frame
    }
    
    if top_score > 0.7:
        pred_boxes = result['pred_boxes'][0, top_idx].unsqueeze(0).cpu()
        keep_class = max_classes[top_idx].item()
        keep_prob = max_probs[top_idx].item()
        
        bboxes = rescale_bboxes(pred_boxes, (width, height))
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox.numpy())
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[keep_class], 3)
            
            # Draw label
            label = f"{CLASSES[keep_class]} {keep_prob:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )
            cv2.rectangle(
                frame, (x1, y1 - text_h - 5), 
                (x1 + text_w, y1), COLORS[keep_class], -1
            )
            cv2.putText(
                frame, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
        
        detection_info.update({
            'detected': True,
            'class': CLASSES[keep_class],
            'confidence': keep_prob,
            'frame': frame
        })
    
    return detection_info

# -------------------------
# Main Application
# -------------------------
def main():
    # Load CSS
    load_css()
    
    # Load Lottie Animation
    lottie_hand = load_lottie_url(
        "https://assets2.lottiefiles.com/packages/lf20_5tl1xxnz.json"
    )
    
    # Hero Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if lottie_hand:
            st_lottie(lottie_hand, height=200, key="hand_animation")
    
    st.markdown(
        '<h1 class="hero-title">🤟 Deteksi Alfabet Bahasa Isyarat Indonesia (BISINDO)</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="hero-quote">"Menjembatani Sunyi dengan Teknologi – Komunikasi Tanpa Batas"</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📚 Tentang Aplikasi")
        st.markdown("""
        <div class="info-card">
            <h3>🔍 Cara Kerja</h3>
            <p>Aplikasi ini menggunakan <b>DETR (Detection Transformer)</b>, 
            arsitektur deep learning modern yang menggabungkan CNN dan 
            Transformer untuk deteksi objek end-to-end.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Fitur Utama</h3>
            <ul>
                <li>Deteksi 26 alfabet BISINDO</li>
                <li>Confidence score</li>
                <li>UI modern & interaktif</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>💡 Tips Penggunaan</h3>
            <ul>
                <li>Pastikan pencahayaan cukup</li>
                <li>Posisikan tangan di tengah</li>
                <li>Jarak ideal 60-80cm dari kamera</li>
                <li>Background kontras lebih baik</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Load Model
    with st.spinner('🔄 Loading AI Model...'):
        model, device = load_model()
        CLASSES, COLORS = get_model_info()
    
    # Device Info
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">🖥️ {device}</div>
        <div class="metric-label">Computing Device</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Content Area
    st.markdown("### 📸 Deteksi Alfabet BISINDO")
    
    # Tab Interface
    tab1, tab2 = st.tabs(["📷 Kamera", "🖼️ Upload Gambar"])
    
    with tab1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### Ambil Foto dari Kamera")
        st.markdown("Arahkan kamera ke gesture tangan BISINDO Anda")
        
        img_file = st.camera_input("📸 Klik untuk mengambil foto", key="camera")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if img_file is not None:
            # Process image
            bytes_data = img_file.read()
            np_img = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            frame = cv2.flip(frame, 1)
            
            # Get prediction
            with st.spinner('🔍 Mendeteksi gesture...'):
                result = process_prediction(frame, model, device, CLASSES, COLORS)
            
            # Display Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📥 Gambar Input")
                st.image(
                    cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 🎯 Hasil Deteksi")
                st.image(
                    cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
            
            # Metrics
            st.markdown("<br>", unsafe_allow_html=True)
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{'✅' if result['detected'] else '❌'}</div>
                    <div class="metric-label">Status Deteksi</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                if result['detected']:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{result['class']}</div>
                        <div class="metric-label">Huruf Terdeteksi</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-container">
                        <div class="metric-value">-</div>
                        <div class="metric-label">Tidak Terdeteksi</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{result['inference_time']:.1f}ms</div>
                    <div class="metric-label">Waktu Inferensi</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence Display
            if result['detected']:
                st.markdown(f"""
                <div class="result-container">
                    <h3 style="color: #E8E8E8; text-align: center;">Confidence Score</h3>
                    <div style="text-align: center;">
                        <span class="prediction-badge">
                            {result['confidence']*100:.1f}% Confidence
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### Upload Gambar dari Galeri")
        st.markdown("Format yang didukung: JPG, JPEG, PNG")
        
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png'],
            key="uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Process uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Get prediction
            with st.spinner('🔍 Mendeteksi gesture...'):
                result = process_prediction(frame, model, device, CLASSES, COLORS)
            
            # Display Results (same as camera tab)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📥 Gambar Input")
                st.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 🎯 Hasil Deteksi")
                st.image(
                    cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
            
            # Metrics (same as camera tab)
            st.markdown("<br>", unsafe_allow_html=True)
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{'✅' if result['detected'] else '❌'}</div>
                    <div class="metric-label">Status Deteksi</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                if result['detected']:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-value">{result['class']}</div>
                        <div class="metric-label">Huruf Terdeteksi</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-container">
                        <div class="metric-value">-</div>
                        <div class="metric-label">Tidak Terdeteksi</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{result['inference_time']:.1f}ms</div>
                    <div class="metric-label">Waktu Inferensi</div>
                </div>
                """, unsafe_allow_html=True)
            
            if result['detected']:
                st.markdown(f"""
                <div class="result-container">
                    <h3 style="color: #E8E8E8; text-align: center;">Confidence Score</h3>
                    <div style="text-align: center;">
                        <span class="prediction-badge">
                            {result['confidence']*100:.1f}% Confidence
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: rgba(232, 232, 232, 0.7); padding: 2rem;">
        <p>🤟 Dibuat dengan ❤️</p>
        <p style="font-size: 0.9rem;">Powered by DETR (Detection Transformer) & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# # ------------------------- DEPLOYMENT CODE STREAMLIT-------------------------
# import streamlit as st
# import cv2
# import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from model import DETR
# from utils.setup import get_classes, get_colors
# from utils.boxes import rescale_bboxes
# import numpy as np
# import time

# # -------------------------
# # Setup
# # -------------------------
# st.title("📸 Deteksi Alfabet BISINDO (Streamlit)")

# # Detect device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# st.text(f"Running on device: {device}")

# # Load model
# model = DETR(num_classes=26)
# model = model.to(device)
# model.eval()
# model.load_pretrained('model/300_model.pt', map_location=device)

# CLASSES = get_classes()
# COLORS = get_colors()

# # Transforms
# transforms = A.Compose([
#     A.Resize(224,224),
#     A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
#     ToTensorV2()
# ])

# # -------------------------
# # Streamlit UI - Camera Input
# # -------------------------
# st.subheader("📸 Ambil gambar dari kamera atau upload file")

# img_file = st.camera_input("Ambil gambar dari kamera")

# if img_file is not None:
#     # Convert file ke OpenCV image
#     bytes_data = img_file.read()
#     np_img = np.frombuffer(bytes_data, np.uint8)
#     frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#     # Mirror
#     frame = cv2.flip(frame, 1)

#     height, width, _ = frame.shape

#     # Preprocess
#     transformed = transforms(image=frame)
#     img_tensor = torch.unsqueeze(transformed['image'], dim=0).to(device)

#     # Inference
#     with torch.no_grad():
#         start_time = time.time()
#         result = model(img_tensor)
#         inference_time = (time.time() - start_time) * 1000  # ms

#     # Post-processing: softmax + Top-1 threshold
#     probabilities = result['pred_logits'].softmax(-1)[0,:,:-1].cpu()
#     max_probs, max_classes = probabilities.max(-1)
#     top_score, top_idx = max_probs.max(0)

#     if top_score > 0.7:
#         pred_boxes = result['pred_boxes'][0, top_idx].unsqueeze(0).cpu()
#         keep_class = max_classes[top_idx].item()
#         keep_prob = max_probs[top_idx].item()

#         bboxes = rescale_bboxes(pred_boxes, (width, height))

#         for bbox in bboxes:
#             x1, y1, x2, y2 = map(int, bbox.numpy())

#             # Draw rectangle
#             cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[keep_class], 3) # pyright: ignore[reportCallIssue]
#             # Draw label
#             label = f"{CLASSES[keep_class]} {keep_prob:.2f}"
#             (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#             cv2.rectangle(frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), COLORS[keep_class], -1) # pyright: ignore[reportCallIssue]
#             cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

#     # Tampilkan hasil
#     st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#              caption=f"Hasil Deteksi | inference {inference_time:.1f} ms",
#              use_container_width=True)



# ------------------------- LOCAL TESTING ONLY -------------------------
# import streamlit as st
# import cv2
# import torch
# import time
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from model import DETR
# from utils.setup import get_classes, get_colors
# from utils.boxes import rescale_bboxes
# import numpy as np

# # -------------------------
# # Setup
# # -------------------------
# st.title("Real-time Sign Language Detection (Streamlit)")

# # Detect device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# st.text(f"Running on device: {device}")

# # Load model
# model = DETR(num_classes=26)
# model = model.to(device)
# model.eval()
# model.load_pretrained('model/300_model.pt', map_location=device)

# CLASSES = get_classes()
# COLORS = get_colors()

# # Transforms
# transforms = A.Compose([
#     A.Resize(224,224),
#     A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
#     ToTensorV2()
# ])

# # -------------------------
# # Streamlit UI
# # -------------------------
# run = st.checkbox("Start Camera")
# FRAME_WINDOW = st.image([])
# fps_text = st.empty()

# cap = cv2.VideoCapture(0)

# # FPS tracking
# frame_count = 0
# fps_start_time = time.time()
# fps_display = 0.0

# try:
#     with torch.no_grad():
#         while run:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to read frame from camera")
#                 break

#             frame = cv2.flip(frame, 1)  # Mirror

#             # Preprocess
#             transformed = transforms(image=frame)
#             img_tensor = torch.unsqueeze(transformed['image'], dim=0).to(device)

#             # Inference
#             inference_start = time.time()
#             result = model(img_tensor)
#             inference_time = (time.time() - inference_start) * 1000  # ms

#             # Post-processing: softmax + Top-1 threshold
#             probabilities = result['pred_logits'].softmax(-1)[0,:,:-1].detach().cpu()
#             max_probs, max_classes = probabilities.max(-1)

#             top_score, top_idx = max_probs.max(0)
#             if top_score > 0.7:
#                 query_indices = top_idx.unsqueeze(0)
#             else:
#                 query_indices = []

#             height, width, _ = frame.shape

#             if len(query_indices) > 0:
#                 pred_boxes = result['pred_boxes'][0, query_indices].detach().cpu()
#                 keep_classes = max_classes[query_indices].detach().cpu()
#                 keep_probs = max_probs[query_indices].detach().cpu()
#                 bboxes = rescale_bboxes(pred_boxes, (width, height))

#                 for bclass, bprob, bbox in zip(keep_classes, keep_probs, bboxes):
#                     x1, y1, x2, y2 = map(int, bbox.numpy())
#                     bclass_idx = int(bclass.numpy())
#                     bprob_val = float(bprob.numpy())

#                     # Draw rectangle
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[bclass_idx], 3)
#                     # Draw label background
#                     (text_w, text_h), baseline = cv2.getTextSize(f"{CLASSES[bclass_idx]} {bprob_val:.2f}",
#                                                                   cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#                     cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), COLORS[bclass_idx], -1)
#                     # Put text
#                     cv2.putText(frame, f"{CLASSES[bclass_idx]} {bprob_val:.2f}", (x1, y1 - 5),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

#             # FPS update
#             frame_count += 1
#             if frame_count % 5 == 0:
#                 elapsed_time = time.time() - fps_start_time
#                 fps_display = 5 / elapsed_time if elapsed_time > 0 else 0
#                 fps_start_time = time.time()

#             # Overlay FPS
#             cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#             # Display
#             FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             fps_text.text(f"FPS: {fps_display:.1f} | Device: {device}")

# except KeyboardInterrupt:
#     st.warning("Interrupted by user")

# finally:
#     cap.release()


# # import streamlit as st
# # import cv2
# # import torch
# # import albumentations as A
# # from model import DETR
# # from utils.setup import get_classes, get_colors
# # from utils.boxes import rescale_bboxes
# # import numpy as np

# # # Load model
# # model = DETR(num_classes=26)
# # model.eval()
# # model.load_pretrained('pretrained/warnet2/aug-new/300_model.pt', map_location=torch.device('cpu'))
# # CLASSES = get_classes()
# # COLORS = get_colors()

# # # Transforms
# # transforms = A.Compose([
# #     A.Resize(224,224),
# #     A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
# #     A.ToTensorV2()
# # ])

# # st.title("Real-time Sign Language Detection")

# # run = st.checkbox("Start Camera")

# # FRAME_WINDOW = st.image([])

# # cap = cv2.VideoCapture(0)

# # while run:
# #     ret, frame = cap.read()
# #     frame = cv2.flip(frame, 1)

# #     transformed = transforms(image=frame)
# #     result = model(torch.unsqueeze(transformed['image'], dim=0))

# #     probabilities = result['pred_logits'].softmax(-1)[:,:,:-1]
# #     max_probs, max_classes = probabilities.max(-1)
# #     keep_mask = max_probs > 0.8

# #     batch_indices, query_indices = torch.where(keep_mask)
# #     height, width, _ = frame.shape
# #     bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (width, height))
# #     classes = max_classes[batch_indices, query_indices]
# #     probas = max_probs[batch_indices, query_indices]

# #     for bclass, bprob, bbox in zip(classes, probas, bboxes):
# #         x1,y1,x2,y2 = bbox.detach().numpy()
# #         cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), COLORS[int(bclass)], 3)
# #         cv2.putText(frame, f"{CLASSES[int(bclass)]} {bprob:.2f}", 
# #                     (int(x1),int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# #     FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# # cap.release()


