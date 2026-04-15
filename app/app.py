import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
from PIL import Image
import time
import io

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Scene Cast AI - Face Segmentation",
    page_icon="🎬",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    def dice_coef(y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        return (2.*K.sum(y_true_f*y_pred_f)+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)
    def dice_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)
    def iou_metric(y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(K.round(y_pred))
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
        return (intersection+smooth)/(union+smooth)
    
    model = tf.keras.models.load_model(
        r"C:\Users\Paramu\guvi_face_segmentation\models\best_model.keras",
        custom_objects={
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'iou_metric': iou_metric
        }
    )
    return model

# ── Predict function ──────────────────────────────────────
def predict_mask(model, image, img_size=128):
    img_resized = cv2.resize(image, (img_size, img_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    start = time.time()
    pred = model.predict(img_input, verbose=0)
    inference_time = (time.time() - start) * 1000
    
    pred_mask = pred[0, :, :, 0]
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    pred_resized = cv2.resize(pred_binary, (image.shape[1], image.shape[0]))
    
    return pred_resized, pred_mask, inference_time

# ── Create overlay ────────────────────────────────────────
def create_overlay(image, mask):
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [255, 0, 0]
    result = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
    
    # Draw bounding boxes around faces
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    face_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result, f'Face {face_count+1}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            face_count += 1
    return result, face_count

# ── Main App ──────────────────────────────────────────────
st.markdown('<p class="main-header">🎬 Scene Cast AI</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:1.2rem;">Real-Time Face Segmentation for Movie Cast Identification</p>', unsafe_allow_html=True)
st.markdown("---")

# Load model
with st.spinner("Loading AI model..."):
    model = load_model()
st.success("✅ Model loaded successfully!")

# Sidebar
st.sidebar.title("⚙️ Settings")
threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)
show_mask = st.sidebar.checkbox("Show Binary Mask", value=True)
show_overlay = st.sidebar.checkbox("Show Overlay", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info("Model: U-Net + MobileNetV2\nTrained on: 327 images\nVal Dice: 0.7139\nVal IoU: 0.5554")

# Tabs
tab1, tab2 = st.tabs(["📸 Image Upload", "📊 Performance Dashboard"])

with tab1:
    st.subheader("Upload a Movie Scene Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        with st.spinner("Detecting faces..."):
            pred_mask, pred_raw, inference_time = predict_mask(model, image)
        
        # Apply threshold from sidebar
        pred_mask_thresh = (cv2.resize(pred_raw, (image.shape[1], image.shape[0])) > threshold).astype(np.uint8)
        overlay_img, face_count = create_overlay(image_rgb, pred_mask_thresh)
        
        # Show metrics
        st.markdown("### 📊 Detection Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Faces Detected", face_count)
        col2.metric("Inference Time", f"{inference_time:.1f} ms")
        col3.metric("Max Confidence", f"{pred_raw.max():.2%}")
        col4.metric("Avg Confidence", f"{pred_raw.mean():.2%}")
        
        st.markdown("---")
        
        # Show images
        cols = [st.columns(3)]
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.image(image_rgb, caption="Original Image", use_container_width=True)
        
        if show_mask:
            with col_b:
                st.image(pred_mask_thresh * 255, caption="Detected Face Mask", 
                        use_container_width=True, clamp=True)
        
        if show_overlay:
            with col_c:
                st.image(overlay_img, caption=f"Overlay ({face_count} faces found)", 
                        use_container_width=True)
        
        # Download log
        st.markdown("---")
        st.subheader("📥 Download Detection Log")
        log_content = f"""Face Segmentation Detection Log
================================
File: {uploaded_file.name}
Faces Detected: {face_count}
Inference Time: {inference_time:.1f} ms
Max Confidence: {pred_raw.max():.4f}
Avg Confidence: {pred_raw.mean():.4f}
Threshold Used: {threshold}
Model: U-Net + MobileNetV2
"""
        st.download_button(
            label="⬇️ Download Log",
            data=log_content,
            file_name="detection_log.txt",
            mime="text/plain"
        )

with tab2:
    st.subheader("📊 Model Performance Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train Dice", "0.9127", "↑ good")
    col2.metric("Val Dice", "0.7139")
    col3.metric("Train IoU", "0.8418", "↑ good")
    col4.metric("Val IoU", "0.5554")
    
    st.markdown("---")
    st.subheader("Training History")
    
    training_img_path = r"C:\Users\Paramu\guvi_face_segmentation\models\training_history.png"
    try:
        st.image(training_img_path, caption="Training History", use_container_width=True)
    except:
        st.info("Training history chart will appear here")
    
    st.markdown("---")
    st.subheader("Sample Predictions")
    predictions_img_path = r"C:\Users\Paramu\guvi_face_segmentation\models\predictions.png"
    try:
        st.image(predictions_img_path, caption="Sample Predictions", use_container_width=True)
    except:
        st.info("Sample predictions will appear here")