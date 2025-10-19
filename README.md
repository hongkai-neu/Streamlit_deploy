# 🚢 Ship Classification Streamlit App

Ocean Hackathon 2025 - Team 01

A machine learning-powered web application that classifies ship types from hydrophone audio recordings.

## 🎯 Features

- **Real-time ship classification** from audio files
- **Four ship types**: Pleasure Craft, Tug, Ferry, Cargo
- **Interactive visualizations**: Waveform, Mel Spectrogram, and Probability charts
- **46 acoustic features** extracted from each audio sample
- **Multiple ML models** support with easy model switching

## 📋 Requirements

See `requirements.txt` for full dependencies. Main libraries:
- streamlit
- librosa
- scikit-learn
- numpy
- pandas
- matplotlib
- scipy
- joblib

## 🚀 Local Deployment

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser:**
   Navigate to `http://localhost:8501`


## 📁 Project Structure

```
Streamlit_deploy/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # Trained ML models
│   ├── VotingClassifier_model.pkl
│   ├── label_encoder.pkl
│   └── ...
└── sample/               # Sample audio files
    ├── 20210105T154808.214Z_class_1_seg_0.wav
    └── ...
```

## 🔧 Configuration

The app uses relative paths for all file operations, making it portable across different environments. Key configuration in `app.py`:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models')
SAMPLE_AUDIO_PATH = os.path.join(BASE_DIR, 'sample')
```

## 🎵 Audio Format

- **Format**: WAV files
- **Sample Rate**: Any (automatically resampled)
- **Duration**: Any (features extracted from full audio)
- **Channels**: Mono or Stereo

## 🧪 Testing

Upload your own WAV files or use the provided sample files to test the classifier.

## 📊 Model Information

The app supports multiple trained models:
- Random Forest
- XGBoost
- Voting Classifier (Ensemble)
- And more...

Select different models from the sidebar to compare performance.

