"""
🚢 Ship Classification Streamlit App
Ocean Hackathon 2025 - Team 01
"""

import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from scipy import signal
from scipy.stats import entropy
import io
import warnings
warnings.filterwarnings('ignore')

# 🎨 Page Configuration
st.set_page_config(
    page_title="Ship Classifier 🚢",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .expected-box {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .match-correct {
        padding: 15px;
        border-radius: 8px;
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
        text-align: center;
        font-weight: bold;
        margin: 15px 0;
    }
    .match-incorrect {
        padding: 15px;
        border-radius: 8px;
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
        text-align: center;
        font-weight: bold;
        margin: 15px 0;
    }
    .confidence-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# 🔹 Configuration - Use relative paths for deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models')
SAMPLE_AUDIO_PATH = os.path.join(BASE_DIR, 'sample')

# 🔹 Helper Functions
def get_available_models():
    """Get list of available model files"""
    if not os.path.exists(MODEL_PATH):
        return []
    
    model_files = [f for f in os.listdir(MODEL_PATH) 
                   if f.endswith('.pkl') and any(keyword in f.lower() 
                   for keyword in ['model', 'forest', 'xgboost', 'bayes', 'ensemble'])]
    return sorted(model_files)

def find_compatible_files(model_name):
    """Find compatible label encoder and scaler for a model"""
    # Use label_encoder.pkl for all models (verified to be identical)
    label_encoder_file = 'label_encoder.pkl'
    if not os.path.exists(os.path.join(MODEL_PATH, label_encoder_file)):
        label_encoder_file = None
    
    # Try to find matching scaler
    scaler_file = None
    for suffix in ['_improved', '', '_feature']:
        for prefix in ['scaler', 'feature_scaler']:
            candidate = f'{prefix}{suffix}.pkl' if suffix else f'{prefix}.pkl'
            if os.path.exists(os.path.join(MODEL_PATH, candidate)):
                scaler_file = candidate
                if 'improved' in model_name and suffix == '_improved':
                    break
                elif 'improved' not in model_name and suffix == '':
                    break
    
    return label_encoder_file, scaler_file

@st.cache_resource
def load_model(model_filename):
    """Load the trained model and auxiliary files"""
    try:
        model_path = os.path.join(MODEL_PATH, model_filename)
        model = joblib.load(model_path)
        
        # Find compatible files
        label_encoder_file, scaler_file = find_compatible_files(model_filename)
        
        label_encoder = None
        scaler = None
        
        if label_encoder_file:
            try:
                label_encoder = joblib.load(os.path.join(MODEL_PATH, label_encoder_file))
            except Exception as e:
                st.warning(f"Could not load label encoder: {e}")
        
        if scaler_file:
            try:
                scaler = joblib.load(os.path.join(MODEL_PATH, scaler_file))
            except Exception as e:
                st.warning(f"Could not load scaler: {e}")
        
        return model, label_encoder, scaler, None
    except Exception as e:
        return None, None, None, str(e)

def compute_spectral_entropy(y, sr):
    """Compute spectral entropy"""
    freqs, psd = signal.welch(y, sr, nperseg=min(256, len(y)))
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]
    return entropy(psd_norm)

def extract_gfcc(y, sr, n_gfcc=13):
    """Extract Gammatone-like features"""
    try:
        D = np.abs(librosa.stft(y))
        n_filters = 40
        filterbank = np.zeros((n_filters, D.shape[0]))
        for i in range(n_filters):
            center = int(i * D.shape[0] / n_filters)
            filterbank[i, max(0, center-5):min(D.shape[0], center+5)] = 1
        filtered = np.dot(filterbank, D)
        log_filtered = np.log(filtered + 1e-10)
        gfcc = librosa.feature.mfcc(S=log_filtered, n_mfcc=n_gfcc)
        return np.mean(gfcc, axis=1)
    except:
        return np.zeros(n_gfcc)

def extract_features(audio_data, sr):
    """Extract all audio features"""
    with st.spinner('🔍 Extracting features...'):
        try:
            # Basic features
            duration = librosa.get_duration(y=audio_data, sr=sr)
            rms = np.mean(librosa.feature.rms(y=audio_data))
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            
            # Spectral features
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
            spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
            spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
            spec_flatness = np.mean(librosa.feature.spectral_flatness(y=audio_data))
            spec_entropy = compute_spectral_entropy(audio_data, sr)
            
            # MFCCs
            mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13), axis=1)
            
            # Chroma
            chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sr), axis=1)
            
            # GFCCs
            gfcc = extract_gfcc(audio_data, sr, n_gfcc=13)
            
            # Combine all features
            features = np.hstack([
                duration, rms, zcr,
                spec_centroid, spec_bandwidth, spec_rolloff,
                spec_flatness, spec_entropy,
                mfcc, chroma, gfcc
            ])
            
            return features
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return None

def predict_ship(features, model, label_encoder, scaler):
    """Make prediction"""
    try:
        # Scale features if scaler available
        if scaler is not None:
            features_scaled = scaler.transform([features])
        else:
            features_scaled = [features]
            st.warning("⚠️ No scaler found - using raw features")
        
        # Predict
        prediction_encoded = model.predict(features_scaled)[0]
        
        # Decode prediction if label encoder available
        if label_encoder is not None:
            prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        else:
            prediction = f"Class {prediction_encoded}"
            st.warning("⚠️ No label encoder found - showing raw class")
        
        # Get probabilities
        try:
            probabilities = model.predict_proba(features_scaled)[0]
            if label_encoder is not None:
                prob_dict = {label_encoder.classes_[i]: prob 
                             for i, prob in enumerate(probabilities)}
            else:
                prob_dict = {f"Class {i}": prob 
                             for i, prob in enumerate(probabilities)}
        except AttributeError:
            # Model doesn't support predict_proba
            prob_dict = {str(prediction): 1.0}
            st.info("ℹ️ Model doesn't support probability predictions")
        
        return prediction, prob_dict
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def plot_waveform(audio_data, sr):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    ax.plot(time, audio_data, color='#1f77b4', linewidth=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_spectrogram(audio_data, sr):
    """Plot mel spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', 
                                   y_axis='mel', ax=ax, cmap='viridis')
    ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

def plot_probabilities(prob_dict):
    """Plot prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(prob_dict.keys())
    probs = [prob_dict[c] * 100 for c in classes]
    colors = ['#2ecc71' if p == max(probs) else '#3498db' for p in probs]
    
    bars = ax.barh(classes, probs, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Confidence (%)', fontsize=12)
    ax.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob + 2, i, f'{prob:.1f}%', 
                va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    return fig

CLASS_NAMES = {
    'class1': 'Pleasure Craft',
    'class2': 'Tug',
    'class3': 'Ferry',
    'class4': 'Cargo'
}

SHIP_EMOJIS = {
    'Pleasure Craft': '⛵',
    'Tug': '🚤',
    'Ferry': '⛴️',
    'Cargo': '🚢'
}

def extract_expected_class_from_filename(filename):
    """
    Extract expected class from filename if it follows the pattern 'class_X'
    Returns the class name (e.g., 'class2') or None if pattern not found
    """
    import re
    match = re.search(r'class_(\d+)', filename.lower())
    if match:
        class_num = match.group(1)
        return f'class{class_num}'
    return None

# 🔹 Main App
def main():
    # Header
    st.markdown('<p class="main-header">🚢 Ship Classification System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ocean Hackathon 2025 | Hydrophone Audio Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ No model files found in the models directory!")
        st.info("Please ensure model files are in the correct directory:")
        st.code(f"{MODEL_PATH}")
        return
    
    # Default to best_model_improved if available
    default_idx = 0
    if 'best_model_improved.pkl' in available_models:
        default_idx = available_models.index('best_model_improved.pkl')
    
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        available_models,
        index=default_idx,
        help="Select which trained model to use for predictions"
    )
    
    # Load model
    with st.spinner('🔄 Loading AI model...'):
        model, label_encoder, scaler, error = load_model(selected_model)
    
    if error:
        st.error(f"❌ Failed to load model: {error}")
        st.info("Please ensure model files are in the correct directory:")
        st.code(f"{MODEL_PATH}")
        return
    
    st.success(f"✅ Model loaded successfully: {selected_model}")
    
    # Model info
    with st.sidebar.expander("📊 Model Information", expanded=False):
        st.write(f"**Model File:** {selected_model}")
        st.write(f"**Model Type:** {type(model).__name__}")
        if label_encoder is not None:
            st.write(f"**Classes:** {', '.join(map(str, label_encoder.classes_))}")
        else:
            st.write("**Classes:** Unknown (no label encoder)")
        st.write(f"**Features:** 46")
        if scaler is not None:
            st.write(f"**Scaler:** ✅ Available")
        else:
            st.write(f"**Scaler:** ❌ Not found")
    
    # Input method selection
    st.sidebar.subheader("📥 Input Method")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Audio File", "Select Sample File"]
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    audio_data = None
    sr = None
    filename = None
    
    # Input handling
    if input_method == "Upload Audio File":
        with col1:
            st.subheader("📤 Upload Audio")
            uploaded_file = st.file_uploader(
                "Choose a WAV file",
                type=['wav'],
                help="Upload a hydrophone recording of ship sounds"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = os.path.join(BASE_DIR, "temp_audio.wav")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Load audio
                audio_data, sr = librosa.load(temp_path, sr=None)
                filename = uploaded_file.name
                
                # Clean up
                os.remove(temp_path)
    
    else:  # Select Sample File
        with col1:
            st.subheader("📁 Select Sample")
            
            # Get list of sample files
            if os.path.exists(SAMPLE_AUDIO_PATH):
                sample_files = [f for f in os.listdir(SAMPLE_AUDIO_PATH) 
                              if f.endswith('.wav')][:20]  # Limit to 20 files
                
                if sample_files:
                    selected_file = st.selectbox(
                        "Choose a sample file:",
                        sample_files,
                        help="Select from available ship recordings"
                    )
                    
                    if st.button("🎵 Load Sample", key="load_sample"):
                        audio_data, sr = librosa.load(
                            os.path.join(SAMPLE_AUDIO_PATH, selected_file),
                            sr=None
                        )
                        filename = selected_file
                else:
                    st.warning("No sample files found in directory")
            else:
                st.warning(f"Sample directory not found: {SAMPLE_AUDIO_PATH}")
                st.info("Please upload a file instead")
    
    # Process and display results
    if audio_data is not None:
        with col1:
            st.success(f"✅ Loaded: {filename}")
            
            # Audio info
            st.write(f"**Duration:** {len(audio_data)/sr:.2f} seconds")
            st.write(f"**Sample Rate:** {sr} Hz")
            st.write(f"**Samples:** {len(audio_data):,}")
            
            # Audio player
            st.audio(audio_data, sample_rate=sr)
        
        # Extract features
        features = extract_features(audio_data, sr)
        
        if features is not None:
            # Make prediction
            with st.spinner('🤖 Analyzing...'):
                prediction, prob_dict = predict_ship(features, model, label_encoder, scaler)
            
            if prediction and prob_dict:
                with col2:
                    st.subheader("🎯 Prediction Results")
                    
                    # Check if filename contains expected class
                    expected_class = extract_expected_class_from_filename(filename) if filename else None
                    expected_ship_type = CLASS_NAMES.get(expected_class, None) if expected_class else None
                    predicted_ship_type = CLASS_NAMES.get(prediction.lower(), prediction)
                    
                    # Display expected result if available
                    if expected_ship_type:
                        col_exp, col_pred = st.columns(2)
                        
                        expected_emoji = SHIP_EMOJIS.get(expected_ship_type, '🚢')
                        predicted_emoji = SHIP_EMOJIS.get(predicted_ship_type, '🚢')
                        
                        with col_exp:
                            st.markdown("**Expected:**")
                            st.markdown(
                                f'<div class="expected-box">{expected_emoji} {expected_ship_type}</div>',
                                unsafe_allow_html=True
                            )
                        
                        with col_pred:
                            st.markdown("**Predicted:**")
                            st.markdown(
                                f'<div class="prediction-box">{predicted_emoji} {predicted_ship_type}</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Show comparison result
                        is_correct = (expected_class.lower() == prediction.lower())
                        if is_correct:
                            st.markdown(
                                '<div class="match-correct">✅ Correct Prediction!</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="match-incorrect">❌ Mismatch - Expected: {expected_ship_type}, Got: {predicted_ship_type}</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        # No expected class - show only prediction
                        predicted_emoji = SHIP_EMOJIS.get(predicted_ship_type, '🚢')
                        st.markdown(
                            f'<div class="prediction-box">{predicted_emoji} {predicted_ship_type}</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Confidence
                    max_prob = max(prob_dict.values()) * 100
                    st.metric("Confidence", f"{max_prob:.1f}%")
                    
                    # Show all probabilities
                    st.write("**All Class Probabilities:**")
                    for ship_class, prob in sorted(prob_dict.items(), 
                                                   key=lambda x: x[1], 
                                                   reverse=True):
                        st.progress(float(prob), text=f"{CLASS_NAMES.get(ship_class.lower(), ship_class)}: {prob*100:.1f}%")
                
                # Visualizations
                st.markdown("---")
                st.subheader("📊 Audio Visualization")
                
                tab1, tab2, tab3 = st.tabs(["🎵 Waveform", "🌈 Spectrogram", "📈 Probabilities"])
                
                with tab1:
                    fig_wave = plot_waveform(audio_data, sr)
                    st.pyplot(fig_wave)
                
                with tab2:
                    fig_spec = plot_spectrogram(audio_data, sr)
                    st.pyplot(fig_spec)
                
                with tab3:
                    fig_prob = plot_probabilities(prob_dict)
                    st.pyplot(fig_prob)
                
                # Feature summary
                with st.expander("🔍 View Extracted Features", expanded=False):
                    feature_names = [
                        'duration', 'rms', 'zcr', 'centroid', 'bandwidth', 
                        'rolloff', 'flatness', 'entropy'
                    ] + [f'mfcc_{i}' for i in range(13)] + \
                        [f'chroma_{i}' for i in range(12)] + \
                        [f'gfcc_{i}' for i in range(13)]
                    
                    feature_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': features
                    })
                    st.dataframe(feature_df, use_container_width=True)
    
    else:
        # Welcome message
        st.info("👆 Please upload an audio file or select a sample to begin analysis")
        
        # Instructions
        with st.expander("ℹ️ How to use this demo", expanded=True):
            st.markdown("""
            ### 📖 Instructions
            
            1. **Upload or Select:** Choose an audio file (WAV format)
            2. **Analyze:** The tool will automatically analyze the ship sounds with the selected model
            3. **View Results:** See the predicted ship type and confidence scores
            4. **Explore:** Check out the waveform and spectrogram visualizations
            
            ### 🚢 Ship Classes
            The model can classify four types of ships:
            - **class1** - Pleasure Craft
            - **class2** - Tug
            - **class3** - Ferry
            - **class4** - Cargo
            
            ### 📊 Features
            The model analyzes 46 acoustic features including:
            - Spectral characteristics (centroid, bandwidth, rolloff)
            - MFCCs (Mel-frequency cepstral coefficients)
            - Chroma features
            - GFCCs (Gammatone frequency cepstral coefficients)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌊 Ocean Hackathon 2025 | Victoria | Team 01 | Ship Classification System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

