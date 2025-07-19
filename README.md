# SVHN Digit Recognition Project

A comprehensive machine learning project for recognizing digits from the Street View House Numbers (SVHN) dataset. This project includes multiple model versions, training scripts, and a web application for real-time digit detection and classification.

## 🏠 Project Overview

This project implements digit recognition using the SVHN dataset with two different model architectures and training approaches:

- **V1**: Basic ShuffleNetV2 model with standard training

- **V2**: Enhanced ShuffleNetV2 model with advanced training techniques (mixed precision, SWA, RandAugment)

- **Web Application**: Streamlit-based interface for real-time digit detection and classification

## 📁 Project Structure

```
SVHN-Digit-Recognition/
├── app.py                 # Main Streamlit web application
├── number_detection.py    # Simple house number detector
├── V1.py                 # Training script for Model V1
├── V2.py                 # Training script for Model V2
├── BestV1.pth           # Pre-trained weights for V1
├── BestV2.pth           # Pre-trained weights for V2
├── V1_metrics.csv       # Training metrics for V1
├── V2_metrics.csv       # Training metrics for V2
├── data/                # SVHN dataset directory
└── README.md           # This file
```

## 🚀 Features

### Model Versions

#### V1 Model
- **Architecture**: ShuffleNetV2 0.5x (lightweight)
- **Training**: Standard training with data augmentation
- **Performance**: ~91.3% validation accuracy
- **Training Time**: 30 epochs

#### V2 Model  
- **Architecture**: ShuffleNetV2 0.5x with advanced training
- **Techniques**: Mixed precision training, SWA (Stochastic Weight Averaging), RandAugment
- **Performance**: ~82.9% validation accuracy (with more aggressive augmentation)
- **Training Time**: 20 epochs

### Web Applications

#### Main Application (`app.py`)

- **OCR + Classification**: Combines EasyOCR for digit detection with custom 
CNN for classification

- **Interactive Interface**: Adjustable confidence thresholds and parameters
- **Multiple Input Sources**: Upload images or use gallery examples
- **Real-time Processing**: Instant digit detection and classification

#### Simple Detector (`number_detection.py`)

- **Basic OCR**: Uses EasyOCR for house number detection
- **Visualization**: Draws bounding boxes around detected numbers
- **Confidence Control**: Adjustable confidence threshold

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- PyTorch
- Streamlit
- OpenCV
- EasyOCR

### Setup

```bash
# Clone the repository
git clone https://github.com/nirvaankohli/SVHN-Digit-Recognition.git
cd SVHN-Digit-Recognition

# Install dependencies
pip install torch torchvision
pip install streamlit opencv-python easyocr
pip install imutils pandas requests

pip install requirements.txt

```

## 📊 Model Performance

### V1 Model Results
- **Final Validation Accuracy**: 91.31%
- **Final Validation Loss**: 0.2835
- **Training Accuracy**: 94.02%

### V2 Model Results  
- **Final Validation Accuracy**: 82.90%
- **Final Validation Loss**: 0.9068
- **Training Accuracy**: 70.80%

*Note: V2 uses more aggressive data augmentation which may reduce training accuracy. I do not have a powerful machine so I can not leave training runing for to long.*

## 🎯 Usage

### Running the Web Application

#### Main Application
```bash
streamlit run app.py
```
Features:
- Upload images or use gallery examples
- Adjust OCR confidence thresholds
- Choose between V1 and V2 models
- Real-time digit detection and classification
- Interactive parameter tuning

#### Simple Detector
```bash
streamlit run number_detection.py
```
Features:
- Basic house number detection
- Adjustable confidence threshold
- Visual bounding box display

### Training Models

#### Train V1 Model
```bash
python V1.py
```

#### Train V2 Model
```bash
python V2.py
```

## 🔧 Configuration

### Model Parameters

#### V1 Training
- **Epochs**: 30
- **Batch Size**: 128
- **Learning Rate**: 0.1
- **Optimizer**: SGD with momentum
- **Data Augmentation**: Random crop, horizontal flip

#### V2 Training
- **Epochs**: 20
- **Batch Size**: 128
- **Learning Rate**: 0.1 (OneCycleLR)
- **Optimizer**: SGD with weight decay
- **Advanced Augmentation**: RandAugment, RandomRotation, RandomErasing
- **Mixed Precision**: Enabled
- **SWA**: Starts at epoch 16

### Web App Parameters
- **Min Confidence**: 0.6 (default)
- **Width Threshold**: 0.7
- **Link Threshold**: 0.4
- **Padding**: Configurable for digits and groups

## 📈 Training Metrics

The training progress for both models is saved in CSV format:
- `V1_metrics.csv`: Complete 30-epoch training history
- `V2_metrics.csv`: 12-epoch training history (SWA implementation)

## 🎨 Key Features

### Advanced Training Techniques (V2)
- **Mixed Precision Training**: Faster training with reduced memory usage
- **Stochastic Weight Averaging (SWA)**: Better generalization
- **RandAugment**: Automated data augmentation
- **OneCycleLR**: Learning rate scheduling

### Web Application Features
- **Multi-model Support**: Switch between V1 and V2
- **Real-time Processing**: Instant results
- **Parameter Tuning**: Interactive sliders for all parameters
- **Gallery Examples**: Pre-loaded test images
- **Export Capabilities**: Save results and annotations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**P.S.** - The models are pre-trained and ready to use. The web application will automatically download the SVHN dataset on first run if not already present.

#### This is part of Shipwrecked

<div align="center">
  <a href="https://shipwrecked.hackclub.com/?t=ghrm" target="_blank">
    <img src="https://hc-cdn.hel1.your-objectstorage.com/s/v3/739361f1d440b17fc9e2f74e49fc185d86cbec14_badge.png" 
         alt="This project is part of Shipwrecked, the world's first hackathon on an island!" 
         style="width: 70%;">
  </a>
</div>