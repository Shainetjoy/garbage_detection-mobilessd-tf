# Garbage Detection - EfficientNet Model

A real-time waste classification system using EfficientNet-B0 for fast and accurate detection of different types of waste materials.

## Features

- 🚀 **Fast Inference** - Optimized for real-time detection (~10-30ms per image)
- 🎯 **High Accuracy** - EfficientNet-B0 model trained on custom dataset
- 📱 **Live Detection** - Webcam support for real-time classification
- 🔍 **7 Waste Categories** - cardboard, clothes, glass, paper, plastic, shoes, trash
- 💾 **Lightweight** - Small model size (~20MB)

## Model Information

- **Architecture**: EfficientNet-B0
- **Classes**: 7 waste categories
- **Input Size**: 224x224
- **Framework**: PyTorch

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Shainetjoy/garbage_detection-mobilessd-tf.git
cd garbage_detection-mobilessd-tf
```

2. Install dependencies:
```bash
pip install -r requirements_efficientdet.txt
```

## Usage

### Live Detection (Webcam)

```bash
python live_detect.py --webcam
```

**Controls:**
- Press `Q` to quit
- Press `S` to save current frame

### Single Image Detection

```bash
python live_detect.py --image path/to/image.jpg
```

### Quick Detection Menu

```bash
python quick_detect.py
```

## Training

To train the model on your own dataset:

```bash
python train_efficientdet.py --data path/to/dataset
```

The dataset should be organized as:
```
dataset/
├── cardboard/
├── clothes/
├── glass/
├── paper/
├── plastic/
├── shoes/
└── trash/
```

## Model Files

- `models/best_model.pth` - Trained model weights
- `models/class_info.json` - Class names and mappings

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- OpenCV 4.8+
- Pillow 9.5+

## Performance

- **Inference Speed**: 10-30ms per image (GPU)
- **Model Size**: ~20MB
- **Accuracy**: Depends on training data

## License

This project is open source and available for use.

## Author

Shainetjoy

## Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch and torchvision teams

