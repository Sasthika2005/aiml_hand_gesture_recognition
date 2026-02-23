# ASL Hand Gesture Recognition

Real-time American Sign Language (ASL) hand gesture recognition system using computer vision and machine learning.

## Features

- Real-time hand gesture detection using MediaPipe
- Recognition of A-Z ASL alphabet signs
- Auto-correction for common words
- Visual feedback with bounding boxes and predictions

## Requirements

```
opencv-python
mediapipe
scikit-learn
numpy
```

## Installation

```bash
pip install opencv-python mediapipe scikit-learn numpy
```

## Usage

1. **Collect training images:**
```bash
python collect_imgs.py
```

2. **Create dataset:**
```bash
python create_dataset.py
```

3. **Train the model:**
```bash
python train_classifier.py
```

4. **Run inference:**
```bash
python inference_classifier.py
```

Press 'q' to quit the camera feed.

## Project Structure

```
├── collect_imgs.py          # Collect hand gesture images
├── create_dataset.py        # Process images and create dataset
├── train_classifier.py      # Train Random Forest classifier
├── inference_classifier.py  # Real-time gesture recognition
├── data/                    # Training images (0-27 folders)
├── data.pickle             # Processed dataset
└── model.p                 # Trained model
```

## How It Works

1. MediaPipe detects hand landmarks (21 points per hand)
2. Landmarks are normalized and converted to features
3. Random Forest classifier predicts the ASL letter
4. Auto-correction suggests common words

## License

MIT
