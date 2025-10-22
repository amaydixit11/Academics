# Quick Exam Setup Script - Run this ONCE before internet is turned off
# chmod +x quick_setup.sh && ./quick_setup.sh

echo "=================================================="
echo "DSP301 EXAM SETUP - ONE-CLICK DOWNLOAD"
echo "=================================================="

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt install -y ffmpeg

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip

# Core packages
pip install torch torchvision torchaudio
pip install transformers sentence-transformers
pip install ultralytics opencv-python mediapipe
pip install librosa soundfile speechrecognition noisereduce jiwer sounddevice scipy
pip install moviepy scikit-learn pandas numpy matplotlib seaborn tqdm wget

# Install Whisper
pip install git+https://github.com/openai/whisper.git

# Create directories
echo "Creating directories..."
mkdir -p models yolo_files temp_audio output test_images test_videos screenshots

# Download YOLO models
echo "Downloading YOLO models..."
python3 -c "
from ultralytics import YOLO
models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8n-seg.pt']
for model in models:
    print(f'Downloading {model}...')
    YOLO(model)
print('YOLO models downloaded!')
"

# Download YOLOv3 files
echo "Downloading YOLOv3 files..."
cd yolo_files
wget -q https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights
wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
wget -q https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
cd ..

# Download Whisper models
echo "Downloading Whisper models..."
python3 -c "
import whisper
models = ['tiny', 'base', 'small']
for model in models:
    print(f'Downloading Whisper {model}...')
    whisper.load_model(model)
print('Whisper models downloaded!')
"

# Download Transformer models
echo "Downloading Transformer models..."
python3 -c "
from transformers import pipeline
from sentence_transformers import SentenceTransformer

print('Downloading BART for zero-shot classification...')
pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

print('Downloading DistilBERT for sentiment analysis...')
pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

print('Downloading Sentence Transformer...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print('Transformer models downloaded!')
"

# Create test files
echo "Creating test files..."
python3 -c "
import cv2
import numpy as np
import soundfile as sf

# Create test image
img = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)
cv2.rectangle(img, (250, 150), (400, 300), (0, 255, 0), -1)
cv2.rectangle(img, (450, 50), (550, 350), (0, 0, 255), -1)
cv2.putText(img, 'Test Image', (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imwrite('test_images/sample.jpg', img)

# Create test audio (sine wave)
sample_rate = 16000
duration = 3.0
frequency = 440
t = np.linspace(0, duration, int(sample_rate * duration))
audio_clean = np.sin(2 * np.pi * frequency * t) * 0.3
audio_noisy = audio_clean + np.random.normal(0, 0.1, len(audio_clean))

sf.write('temp_audio/clean_sample.wav', audio_clean, sample_rate)
sf.write('temp_audio/noisy_sample.wav', audio_noisy, sample_rate)

# Create test video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_videos/sample_video.avi', fourcc, 10.0, (640, 480))

for i in range(30):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    x = int(50 + i * 15)
    y = int(200 + 50 * np.sin(i * 0.3))
    cv2.rectangle(frame, (x, y), (x+100, y+80), (0, 255, 0), -1)
    cv2.putText(frame, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out.write(frame)

out.release()
print('Test files created successfully!')
"

# Verify installation
echo "Verifying installation..."
python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
except: print('PyTorch: NOT INSTALLED')

try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
except: print('OpenCV: NOT INSTALLED')

try:
    import whisper
    print('Whisper: INSTALLED')
except: print('Whisper: NOT INSTALLED')

try:
    from ultralytics import YOLO
    print('YOLOv8: INSTALLED')
except: print('YOLOv8: NOT INSTALLED')

try:
    import mediapipe as mp
    print(f'MediaPipe: {mp.__version__}')
except: print('MediaPipe: NOT INSTALLED')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except: print('Transformers: NOT INSTALLED')

import os
print('\nDownloaded model files:')
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith(('.pt', '.weights', '.cfg', '.names')):
            print(f'  {os.path.join(root, file)}')
"

echo "=================================================="
echo "SETUP COMPLETE!"
echo "=================================================="
echo "All models and dependencies downloaded successfully!"
echo "You can now run your exam notebook OFFLINE."
echo "Files saved in:"
echo "  - ~/.cache/huggingface/ (Transformer models)"
echo "  - ~/.cache/torch/ (YOLO models)"  
echo "  - yolo_files/ (YOLOv3 files)"
echo "  - test_images/, temp_audio/, test_videos/ (test files)"
echo "=================================================="