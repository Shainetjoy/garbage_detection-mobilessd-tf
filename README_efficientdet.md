# EfficientNet Waste Classification Model

This project uses **EfficientNet-B0** for fast, accurate waste classification optimized for live detection. The model is trained on your custom dataset with 7 classes: cardboard, clothes, glass, paper, plastic, shoes, and trash.

## Why EfficientNet?

- **Fast inference** - Optimized for real-time detection
- **High accuracy** - State-of-the-art classification performance
- **Lightweight** - Smaller model size than YOLO
- **Mobile-friendly** - Can be easily converted to TensorFlow Lite for mobile deployment

## Dataset Structure

Your dataset should be organized as:
```
Try/dataset/
├── cardboard/
│   └── *.jpg
├── clothes/
│   └── *.jpg
├── glass/
│   └── *.jpg
├── paper/
│   └── *.jpg
├── plastic/
│   └── *.jpg
├── shoes/
│   └── *.jpg
└── trash/
    └── *.jpg
```

## Setup

1. **Activate your virtual environment:**
   ```bash
   garbage_tf_env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements_efficientdet.txt
   ```

## Training

Train the model on your dataset:

```bash
python train_efficientdet.py --data D:/garbage_detection/Try/dataset
```

### Training Options

```bash
# Custom epochs and batch size
python train_efficientdet.py --data D:/garbage_detection/Try/dataset --epochs 100 --batch-size 64

# Use CPU instead of GPU
python train_efficientdet.py --data D:/garbage_detection/Try/dataset --device cpu

# Custom learning rate
python train_efficientdet.py --data D:/garbage_detection/Try/dataset --lr 0.0001
```

### What Happens During Training

1. **Automatic Train/Val Split**: The script automatically splits your dataset into 80% training and 20% validation
2. **Data Augmentation**: Random flips, rotations, and color jitter for better generalization
3. **Model Checkpoints**: Best model is saved automatically, plus checkpoints every 10 epochs
4. **Progress Tracking**: Real-time loss and accuracy metrics

### Output Files

After training, you'll find:
- `models/best_model.pth` - Best model based on validation accuracy
- `models/checkpoint_epoch_N.pth` - Checkpoints every 10 epochs
- `models/class_info.json` - Class names and mappings
- `train/` and `val/` folders - Automatically created train/validation split

## Live Detection

### Single Image Prediction

```bash
python live_detect.py --image path/to/image.jpg
```

### Webcam Live Feed

```bash
python live_detect.py --webcam
```

**Webcam Controls:**
- Press `Q` to quit
- Press `S` to save current frame

### Detection Options

```bash
# Custom model path
python live_detect.py --image image.jpg --model models/best_model.pth

# Use CPU
python live_detect.py --webcam --device cpu

# Custom confidence threshold for webcam
python live_detect.py --webcam --conf 0.7

# Don't show window
python live_detect.py --image image.jpg --no-show
```

## Model Performance

- **Inference Speed**: ~10-30ms per image (on GPU)
- **Model Size**: ~20MB
- **Accuracy**: Depends on your dataset and training epochs

## Tips for Better Results

1. **More Training**: Increase epochs for better accuracy
   ```bash
   python train_efficientdet.py --data D:/garbage_detection/Try/dataset --epochs 100
   ```

2. **Larger Batch Size**: If you have enough GPU memory
   ```bash
   python train_efficientdet.py --data D:/garbage_detection/Try/dataset --batch-size 64
   ```

3. **Data Quality**: Ensure your images are clear and well-labeled

4. **Class Balance**: If classes are imbalanced, the model may favor majority classes

## Troubleshooting

1. **Out of Memory Error**:
   - Reduce batch size: `--batch-size 16`
   - Use CPU: `--device cpu`

2. **Low Accuracy**:
   - Train for more epochs
   - Check data quality
   - Ensure balanced dataset

3. **Slow Inference**:
   - Use GPU: `--device cuda`
   - Reduce image resolution (modify transform in code)

## Comparison with YOLO

| Feature | EfficientNet | YOLO |
|---------|-------------|------|
| Speed | Fast | Very Fast |
| Accuracy | High | High |
| Model Size | Small (~20MB) | Medium (~50MB) |
| Use Case | Classification | Object Detection |
| Live Detection | Excellent | Excellent |

For classification tasks (like yours), EfficientNet is often faster and more accurate than YOLO.

## Next Steps

1. Train the model: `python train_efficientdet.py --data D:/garbage_detection/Try/dataset`
2. Test on images: `python live_detect.py --image path/to/image.jpg`
3. Use webcam: `python live_detect.py --webcam`

Enjoy your fast, accurate waste classification model!

