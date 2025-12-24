import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import plotly.graph_objects as go
import plotly.express as px
import os

# Page Configuration
st.set_page_config(
    page_title="Klasifikasi Kematangan Tomat",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk background merah profesional
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #8B0000 0%, #DC143C 50%, #8B0000 100%);
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .model-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 10px 0;
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        margin: 20px 0;
    }
    
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: white !important;
    }
    
    .stSelectbox label, .stFileUploader label {
        color: white !important;
        font-weight: bold;
        font-size: 18px;
    }
    
    .uploadedFile {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    div[data-testid="stMetricValue"] {
        color: white !important;
        font-size: 28px;
    }
    
    div[data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    .success-box {
        background: rgba(34, 197, 94, 0.2);
        border: 2px solid rgba(34, 197, 94, 0.5);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .info-box {
        background: rgba(59, 130, 246, 0.2);
        border: 2px solid rgba(59, 130, 246, 0.5);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: rgba(251, 146, 60, 0.2);
        border: 2px solid rgba(251, 146, 60, 0.5);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Model Information
MODELS_INFO = {
    'CNN Model': {
        'accuracy': '76%',
        'description': 'Custom Convolutional Neural Network',
        'performance': {
            'Damaged': {'precision': 0.86, 'recall': 0.05, 'f1': 0.09},
            'Old': {'precision': 0.64, 'recall': 0.89, 'f1': 0.74},
            'Ripe': {'precision': 0.81, 'recall': 0.70, 'f1': 0.75},
            'Unripe': {'precision': 0.97, 'recall': 1.00, 'f1': 0.98}
        }
    },
    'ResNet50': {
        'accuracy': '97%',
        'description': 'Deep Residual Network with Transfer Learning',
        'performance': {
            'Damaged': {'precision': 0.98, 'recall': 0.85, 'f1': 0.91},
            'Old': {'precision': 0.95, 'recall': 0.98, 'f1': 0.96},
            'Ripe': {'precision': 0.98, 'recall': 0.97, 'f1': 0.98},
            'Unripe': {'precision': 0.98, 'recall': 1.00, 'f1': 0.99}
        }
    },
    'EfficientNetB0': {
        'accuracy': '98%',
        'description': 'State-of-the-art Efficient Network Architecture',
        'performance': {
            'Damaged': {'precision': 0.98, 'recall': 0.92, 'f1': 0.95},
            'Old': {'precision': 0.97, 'recall': 0.98, 'f1': 0.98},
            'Ripe': {'precision': 0.99, 'recall': 0.99, 'f1': 0.99},
            'Unripe': {'precision': 0.99, 'recall': 1.00, 'f1': 1.00}
        }
    }
}

CLASS_INFO = {
    'Damaged': {
        'icon': '🔴',
        'color': '#DC2626',
        'description': 'Tomat rusak atau tidak layak konsumsi'
    },
    'Old': {
        'icon': '🟠',
        'color': '#EA580C',
        'description': 'Tomat terlalu matang atau sudah tua'
    },
    'Ripe': {
        'icon': '🟢',
        'color': '#16A34A',
        'description': 'Tomat matang sempurna dan siap dikonsumsi'
    },
    'Unripe': {
        'icon': '🟡',
        'color': '#CA8A04',
        'description': 'Tomat masih mentah atau belum matang'
    }
}

CLASS_NAMES = ['Damaged', 'Old', 'Ripe', 'Unripe']

# Initialize session state
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {}

# Functions
@st.cache_resource
def load_model_from_file(model_file, model_name):
    """Load model from uploaded file"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{model_name}.h5"
        with open(temp_path, "wb") as f:
            f.write(model_file.getbuffer())
        
        # Load model
        model = load_model(temp_path, compile=False)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_and_preprocess_image(img, target_size=(224, 224)):
    """Load and preprocess image for prediction"""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_image(model, img_array):
    """Make prediction on preprocessed image"""
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    # Get all class probabilities
    class_probabilities = {
        CLASS_NAMES[i]: float(predictions[0][i]) 
        for i in range(len(CLASS_NAMES))
    }
    
    return predicted_class, confidence, class_probabilities

def create_probability_chart(probabilities):
    """Create horizontal bar chart for class probabilities"""
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = [CLASS_INFO[cls]['color'] for cls in classes]
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{p*100:.1f}%' for p in probs],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Probabilitas Setiap Kelas",
        xaxis_title="Probabilitas",
        yaxis_title="Kelas",
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.1)',
        font=dict(color='white', size=12),
        xaxis=dict(range=[0, 1]),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_performance_table(model_name):
    """Create performance metrics table"""
    performance = MODELS_INFO[model_name]['performance']
    
    data = []
    for class_name, metrics in performance.items():
        data.append({
            'Kelas': f"{CLASS_INFO[class_name]['icon']} {class_name}",
            'Precision': f"{metrics['precision']*100:.0f}%",
            'Recall': f"{metrics['recall']*100:.0f}%",
            'F1-Score': f"{metrics['f1']*100:.0f}%"
        })
    
    df = pd.DataFrame(data)
    return df

# Header
st.markdown("""
    <div class="main-header">
        <h1 style='font-size: 48px; margin-bottom: 10px;'>🍅 Sistem Klasifikasi Kematangan Tomat</h1>
        <p style='font-size: 20px; opacity: 0.9;'>Deteksi tingkat kematangan tomat menggunakan Deep Learning</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 Pilih Model AI")
    
    selected_model = st.selectbox(
        "Model",
        options=list(MODELS_INFO.keys()),
        label_visibility="collapsed"
    )
    
    st.markdown(f"""
        <div class="model-card">
            <h4>{selected_model}</h4>
            <p style='font-size: 14px; opacity: 0.9;'>{MODELS_INFO[selected_model]['description']}</p>
            <hr style='border-color: rgba(255,255,255,0.2);'>
            <p><strong>Akurasi:</strong> <span style='font-size: 24px;'>{MODELS_INFO[selected_model]['accuracy']}</span></p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload Model Section
    st.markdown("### 📁 Upload Model File")
    
    model_file = st.file_uploader(
        f"Upload {selected_model} (.h5)",
        type=['h5'],
        key=f"model_uploader_{selected_model}",
        help="Upload file model .h5 yang telah Anda latih"
    )
    
    if model_file is not None:
        if selected_model not in st.session_state.loaded_models:
            with st.spinner(f'Loading {selected_model}...'):
                loaded_model = load_model_from_file(model_file, selected_model)
                if loaded_model is not None:
                    st.session_state.loaded_models[selected_model] = loaded_model
                    st.success(f"✅ {selected_model} berhasil dimuat!")
        else:
            st.success(f"✅ {selected_model} sudah dimuat!")
    else:
        if selected_model in st.session_state.loaded_models:
            del st.session_state.loaded_models[selected_model]
        st.markdown("""
            <div class="warning-box">
                <p style='font-size: 12px;'><strong>⚠️ Model belum dimuat</strong></p>
                <p style='font-size: 11px; opacity: 0.8;'>Upload file model untuk melakukan prediksi</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Informasi Dataset")
    st.markdown("""
        <div class="info-box">
            <p><strong>Total Gambar:</strong> 6,487</p>
            <p><strong>Training:</strong> 4,150</p>
            <p><strong>Validation:</strong> 1,038</p>
            <p><strong>Testing:</strong> 1,298</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Kategori Klasifikasi")
    for class_name, info in CLASS_INFO.items():
        st.markdown(f"""
            <div class="info-box">
                <p style='font-size: 16px;'><strong>{info['icon']} {class_name}</strong></p>
                <p style='font-size: 12px; opacity: 0.8;'>{info['description']}</p>
            </div>
        """, unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("### 📤 Upload Gambar Tomat")
    
    uploaded_file = st.file_uploader(
        "Pilih gambar tomat (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Gambar yang di-upload', use_container_width=True)
        
        # Check if model is loaded
        model_loaded = selected_model in st.session_state.loaded_models
        
        # Predict button
        predict_button = st.button(
            "🔍 Analisis Gambar", 
            use_container_width=True, 
            type="primary",
            disabled=not model_loaded
        )
        
        if not model_loaded:
            st.warning("⚠️ Silakan upload model terlebih dahulu di sidebar!")
        
        if predict_button and model_loaded:
            with st.spinner('Memproses gambar...'):
                try:
                    # Preprocess image
                    img_array = load_and_preprocess_image(img)
                    
                    # Get loaded model
                    model = st.session_state.loaded_models[selected_model]
                    
                    # Make prediction
                    predicted_class, confidence, class_probabilities = predict_image(model, img_array)
                    
                    # Store in session state
                    st.session_state.prediction = predicted_class
                    st.session_state.confidence = confidence
                    st.session_state.probabilities = class_probabilities
                    st.session_state.model_used = selected_model
                    
                    st.success("✅ Prediksi berhasil!")
                    
                except Exception as e:
                    st.error(f"❌ Error saat prediksi: {str(e)}")

with col2:
    if 'prediction' in st.session_state:
        st.markdown("### ✅ Hasil Prediksi")
        
        predicted_class = st.session_state.prediction
        confidence = st.session_state.confidence
        probabilities = st.session_state.probabilities
        
        # Result card
        icon = CLASS_INFO[predicted_class]['icon']
        color = CLASS_INFO[predicted_class]['color']
        description = CLASS_INFO[predicted_class]['description']
        
        st.markdown(f"""
            <div class="result-card" style='background: linear-gradient(135deg, {color}40 0%, {color}20 100%); border-color: {color};'>
                <div style='text-align: center;'>
                    <div style='font-size: 80px; margin-bottom: 20px;'>{icon}</div>
                    <h2 style='font-size: 42px; margin-bottom: 10px;'>{predicted_class}</h2>
                    <p style='font-size: 18px; opacity: 0.9; margin-bottom: 20px;'>{description}</p>
                    <div style='background: rgba(255,255,255,0.2); border-radius: 15px; padding: 20px; display: inline-block;'>
                        <p style='font-size: 14px; opacity: 0.8; margin-bottom: 5px;'>Tingkat Kepercayaan</p>
                        <p style='font-size: 36px; font-weight: bold;'>{confidence*100:.1f}%</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Model", st.session_state.model_used)
        with col_m2:
            st.metric("Akurasi Model", MODELS_INFO[st.session_state.model_used]['accuracy'])
        with col_m3:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        
        # Probability chart
        st.markdown("### 📊 Detail Probabilitas")
        fig = create_probability_chart(probabilities)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show all probabilities in table format
        st.markdown("### 📋 Probabilitas Detail")
        prob_data = []
        for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            prob_data.append({
                'Kelas': f"{CLASS_INFO[cls]['icon']} {cls}",
                'Probabilitas': f"{prob*100:.2f}%",
                'Confidence': prob
            })
        
        prob_df = pd.DataFrame(prob_data)
        st.dataframe(
            prob_df[['Kelas', 'Probabilitas']], 
            use_container_width=True, 
            hide_index=True
        )
        
    else:
        st.markdown("""
            <div class="info-box" style='text-align: center; padding: 60px 20px;'>
                <h3>📸 Upload gambar untuk memulai analisis</h3>
                <p style='opacity: 0.8;'>Pilih gambar tomat dari komputer Anda dan klik tombol "Analisis Gambar"</p>
            </div>
        """, unsafe_allow_html=True)

# Performance Metrics Section
st.markdown("---")
st.markdown(f"### 📈 Performa Model {selected_model}")

df_performance = create_performance_table(selected_model)

# Style the dataframe
st.markdown("""
    <style>
    .dataframe {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    .dataframe td, .dataframe th {
        color: white !important;
        text-align: center !important;
    }
    </style>
""", unsafe_allow_html=True)

st.dataframe(df_performance, use_container_width=True, hide_index=True)

# Model Status Section
st.markdown("---")
st.markdown("### 🔧 Status Model")

status_cols = st.columns(3)
for idx, (model_name, _) in enumerate(MODELS_INFO.items()):
    with status_cols[idx]:
        if model_name in st.session_state.loaded_models:
            st.markdown(f"""
                <div class="success-box" style='text-align: center;'>
                    <p style='font-size: 16px;'><strong>✅ {model_name}</strong></p>
                    <p style='font-size: 12px; opacity: 0.8;'>Model Loaded</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="warning-box" style='text-align: center;'>
                    <p style='font-size: 16px;'><strong>⚠️ {model_name}</strong></p>
                    <p style='font-size: 12px; opacity: 0.8;'>Not Loaded</p>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; opacity: 0.8; padding: 20px;'>
        <p>🚀 Powered by Deep Learning • CNN, ResNet50, EfficientNetB0</p>
        <p style='font-size: 12px;'>© 2024 Tomato Classification System</p>
    </div>
""", unsafe_allow_html=True)