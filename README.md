# 🚢 Ship Classification Project - Ocean Hackathon 2025

**Team 01 - Victoria**

A complete machine learning project for classifying ship types from hydrophone audio recordings, including exploratory data analysis, model development, and an interactive web application deployment.

## 📚 Project Overview

This repository contains our submission for the Ocean Hackathon 2025, featuring:
- **Data Analysis & Exploration** - Jupyter notebooks with Exploratory Data Analysis and Model Training
- **Interactive Web App** - Streamlit-based deployment for real-time predictions

## 🎯 Ship Classification

The system classifies four types of ships from hydrophone audio:
- **Class 1**: Pleasure Craft ⛵
- **Class 2**: Tug 🚤
- **Class 3**: Ferry ⛴️
- **Class 4**: Cargo 🚢

Using 46 acoustic features including spectral characteristics, MFCCs, chroma, and GFCCs.

## 📁 Project Structure

```
Streamlit_deploy/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
│
├── notebook/                          # 📊 Analysis & Development
│   ├── hackathon_eda.ipynb           # Exploratory data analysis
│   └── hackathon_modeling.ipynb      # Model training & evaluation
│
├── models/                            # 🤖 Selected Trained Models for prototype
│   ├── label_encoder.pkl             # Class label encoder
│   ├── random_forest.pkl             # Random Forest classifier
│   └── xgboost_tuned.pkl            # Tuned XGBoost model
│
└── sample/                            # 🎵 Sample Audio Files
    ├── 20210105T154808.214Z_class_1_seg_0.wav
    ├── 20210114T024644.264Z_class_2_seg_0.wav
    └── 20210118T032549.783Z_class_2_seg_0.wav
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Streamlit_deploy

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### 3. Explore the Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Open the notebooks in the notebook/ directory:
# - hackathon_eda.ipynb: Data exploration and feature analysis
# - hackathon_modeling.ipynb: Model training, tuning, and evaluation
```

## 🎨 Web Application Features

- **Real-time Classification**: Upload or select audio files for instant predictions
- **Interactive Visualizations**: 
  - Audio waveform display
  - Mel spectrogram analysis
  - Probability distribution charts
- **Model Selection**: Compare different trained models
- **Confidence Scores**: View prediction probabilities for all classes
- **Feature Extraction**: 46 acoustic features automatically extracted
- **Ground Truth Comparison**: Automatic validation with sample files

## 📊 Notebooks

### `hackathon_eda.ipynb`
Exploratory Data Analysis including:
- Audio data distribution and characteristics
- Feature correlation analysis
- Class balance investigation
- Spectral analysis and visualization

### `hackathon_modeling.ipynb`
Model Development including:
- Feature engineering pipeline
- Multiple classifier training (Random Forest, XGBoost, etc.)
- Hyperparameter tuning
- Model evaluation and comparison
- Final model selection

## 🤖 Models

The `models/` directory contains trained classifiers:

- **Random Forest** (`random_forest.pkl`): Ensemble decision tree classifier
- **XGBoost** (`xgboost_tuned.pkl`): Gradient boosted model with tuned hyperparameters
- **Label Encoder** (`label_encoder.pkl`): Converts between class names and numeric labels

All models are trained on acoustic features extracted from hydrophone recordings.

## 📋 Requirements

Key dependencies:
- **streamlit** - Web application framework
- **librosa** - Audio analysis and feature extraction
- **scikit-learn** - Machine learning models
- **xgboost** - Gradient boosting framework
- **numpy, pandas** - Data manipulation
- **matplotlib** - Visualizations
- **scipy** - Signal processing

See `requirements.txt` for complete list with versions.

## 🎵 Audio Format

- **Format**: WAV files
- **Sample Rate**: Any (automatically resampled)
- **Duration**: Any (features extracted from full audio)
- **Channels**: Mono or Stereo supported

## 🧪 Testing the Application

1. Use the provided sample files in the `sample/` directory
2. Upload your own hydrophone recordings (WAV format)
3. Compare predictions against expected classes (filenames with `class_X` pattern)

## 📈 Model Performance

For detailed metrics, evaluation results, and comparison between models, please refer to the `hackathon_modeling.ipynb` notebook.

## 🔧 Configuration

The application uses relative paths for portability:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models')
SAMPLE_AUDIO_PATH = os.path.join(BASE_DIR, 'sample')
```
