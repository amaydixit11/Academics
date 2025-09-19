# DSP301 Exam Day Checklist & Quick Commands

## 🚀 PRE-EXAM SETUP (Run with Internet)

### Step 1: Download Everything
```bash
# Run the setup script
chmod +x quick_setup.sh
./quick_setup.sh
```

### Step 2: Verify Downloads
```python
# Quick verification
import torch, cv2, whisper, transformers
from ultralytics import YOLO
print("All libraries loaded successfully!")
```

## 📋 EXAM DAY QUICK REFERENCE

### Assignment 1: Text Processing
```python
# BERT Classification
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
result = classifier("I love this!")

# Zero-shot Classification  
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier("Turn on lights", ["smart_home", "music", "weather"])

# Tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("Hello world")
```

### Assignment 2: Audio Processing
```python
# Whisper ASR
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.wav")
print(result["text"])

# Noise Reduction
import noisereduce as nr
import librosa
y, sr = librosa.load("noisy_audio.wav")
reduced = nr.reduce_noise(y=y, sr=sr)

# WER Calculation
from jiwer import wer
error_rate = wer("reference text", "hypothesis text") * 100
```

### Assignment 3: Computer Vision
```python
# YOLOv8 Detection
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('image.jpg')
results[0].show()

# Basic OpenCV
import cv2
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# YOLOv3 (if needed)
net = cv2.dnn.readNetFromDarknet('yolo_files/yolov3.cfg', 'yolo_files/yolov3.weights')
```

### Assignment 4: Real-time CV
```python
# Webcam detection
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow('Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
```

### Assignment 5: Multimodal
```python
# Complete pipeline
import whisper
import mediapipe as mp
from transformers import pipeline
from moviepy.editor import VideoFileClip

# Audio processing
whisper_model = whisper.load_model("base")
text_result = whisper_model.transcribe("audio.wav")

# Hand gesture detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Intent classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
intent = classifier(text_result["text"], ["forward", "left", "right", "stop"])
```

## 🔧 Common Commands

### File Operations
```python
import os
os.makedirs("output", exist_ok=True)
os.path.exists("file.txt")
```

### Image Processing
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read/Save
img = cv2.imread('image.jpg')
cv2.imwrite('output.jpg', img)

# Color conversion
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

### Audio Processing
```python
import librosa
import soundfile as sf

# Load audio
y, sr = librosa.load('audio.wav', sr=16000)

# Save audio
sf.write('output.wav', y, sr)

# Features
mfcc = librosa.feature.mfcc(y=y, sr=sr)
```

## ⚠️ Common Errors & Fixes

### CUDA/Memory Issues
```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()  # Clear CUDA memory
```

### Model Loading Issues
```python
# Use CPU if CUDA fails
model = YOLO('yolov8n.pt').to('cpu')
pipeline("...", device=-1)  # Force CPU
```

### File Path Issues
```python
import os
print(os.getcwd())  # Check current directory
os.listdir('.')     # List files
```

## 📁 Expected File Structure
```
project/
├── yolo_files/
│   ├── yolov3.weights
│   ├── yolov3.cfg
│   └── coco.names
├── temp_audio/
├── output/
├── test_images/
├── test_videos/
├── ultimate_notebook.ipynb
└── quick_setup.sh
```

## 🎯 Exam Tips

1. **Test imports first**: Run all import statements before starting
2. **Use sample data**: Create test files if uploads fail
3. **Check file paths**: Use absolute paths if relative paths fail
4. **Monitor memory**: Use smaller models if memory issues occur
5. **Save frequently**: Save outputs to files for verification
6. **Have backups**: Keep multiple versions of working code

## 🚨 Emergency Code Snippets

### Create Test Image
```python
import cv2, numpy as np
img = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)
cv2.imwrite('test.jpg', img)
```

### Create Test Audio
```python
import numpy as np, soundfile as sf
sr, duration = 16000, 2.0
t = np.linspace(0, duration, int(sr * duration))
audio = np.sin(2 * np.pi * 440 * t) * 0.3
sf.write('test.wav', audio, sr)
```

### Basic Video
```python
import cv2, numpy as np
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test.avi', fourcc, 10.0, (640, 480))
for i in range(30):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out.write(frame)
out.release()
```

## ✅ Final Checklist

- [ ] All libraries installed and tested
- [ ] Models downloaded and verified
- [ ] Test files created
- [ ] Sample code runs without errors
- [ ] Output directories created
- [ ] Notebook ready to run offline

**Good luck with your exam!** 🚀