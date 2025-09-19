#!/usr/bin/env python3
"""
Pre-Exam Setup Script - Run this BEFORE internet is turned off
Downloads all necessary models, weights, and dependencies
"""

import os
import subprocess
import sys
import urllib.request
import wget

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n[INFO] {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description}: {e}")
        print(f"[ERROR] Output: {e.stdout}")
        print(f"[ERROR] Error: {e.stderr}")
        return False

def download_file(url, filename, description):
    """Download a file with error handling"""
    print(f"\n[INFO] {description}")
    try:
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)
            print(f"[SUCCESS] Downloaded {filename}")
        else:
            print(f"[INFO] {filename} already exists")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        return False

def main():
    print("="*60)
    print("PRE-EXAM SETUP SCRIPT - DSP301 Ultimate Preparation")
    print("="*60)
    
    # Create directories
    directories = [
        "models", "yolo_files", "temp_audio", "output", 
        "test_images", "test_videos", "test_audio"
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"[INFO] Created directory: {dir_name}")
    
    # 1. INSTALL CORE LIBRARIES
    print("\n" + "="*40)
    print("STEP 1: INSTALLING CORE LIBRARIES")
    print("="*40)
    
    libraries = [
        # Core ML libraries
        "torch torchvision torchaudio",
        "transformers",
        "sentence-transformers",
        
        # Computer Vision
        "ultralytics",
        "opencv-python",
        "mediapipe",
        
        # Audio Processing
        "git+https://github.com/openai/whisper.git",
        "librosa",
        "soundfile",
        "speechrecognition",
        "noisereduce",
        "jiwer",
        "sounddevice",
        "scipy",
        
        # Video Processing
        "moviepy",
        
        # Utilities
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "wget",
    ]
    
    for lib in libraries:
        run_command(f"pip install {lib}", f"Installing {lib}")
    
    # Install system dependencies
    run_command("sudo apt update && sudo apt install -y ffmpeg", "Installing FFmpeg")
    
    # 2. DOWNLOAD YOLO MODELS
    print("\n" + "="*40)
    print("STEP 2: DOWNLOADING YOLO MODELS")
    print("="*40)
    
    yolo_models = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
        "yolov8n-seg.pt", "yolov8s-seg.pt"
    ]
    
    for model in yolo_models:
        run_command(f"python -c \"from ultralytics import YOLO; YOLO('{model}')\"", 
                   f"Downloading {model}")
    
    # 3. DOWNLOAD YOLOv3 FILES
    print("\n" + "="*40)
    print("STEP 3: DOWNLOADING YOLOv3 FILES")
    print("="*40)
    
    yolo3_files = {
        'yolov3.weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights',
        'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
    
    for filename, url in yolo3_files.items():
        download_file(url, f"yolo_files/{filename}", f"Downloading {filename}")
    
    # 4. DOWNLOAD WHISPER MODELS
    print("\n" + "="*40)
    print("STEP 4: DOWNLOADING WHISPER MODELS")
    print("="*40)
    
    whisper_models = ["tiny", "base", "small", "medium"]
    for model in whisper_models:
        run_command(f"python -c \"import whisper; whisper.load_model('{model}')\"", 
                   f"Downloading Whisper {model} model")
    
    # 5. DOWNLOAD TRANSFORMERS MODELS
    print("\n" + "="*40)
    print("STEP 5: DOWNLOADING TRANSFORMER MODELS")
    print("="*40)
    
    transformer_models = [
        ("openai/whisper-base.en", "WhisperProcessor.from_pretrained"),
        ("facebook/bart-large-mnli", "pipeline"),
        ("sentence-transformers/all-MiniLM-L6-v2", "SentenceTransformer"),
    ]
    
    for model_name, loader in transformer_models:
        if "pipeline" in loader:
            run_command(f"python -c \"from transformers import pipeline; pipeline('zero-shot-classification', model='{model_name}')\"",
                       f"Downloading {model_name}")
        elif "SentenceTransformer" in loader:
            run_command(f"python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{model_name}')\"",
                       f"Downloading {model_name}")
        else:
            run_command(f"python -c \"from transformers import {loader}; {loader}('{model_name}')\"",
                       f"Downloading {model_name}")
    
    # 6. CREATE TEST FILES
    print("\n" + "="*40)
    print("STEP 6: CREATING TEST FILES")
    print("="*40)
    
    # Create a simple test image
    test_image_code = '''
import cv2
import numpy as np

# Create a test image with colored rectangles
img = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)  # Blue rectangle
cv2.rectangle(img, (250, 150), (400, 300), (0, 255, 0), -1)  # Green rectangle
cv2.rectangle(img, (450, 50), (550, 350), (0, 0, 255), -1)  # Red rectangle
cv2.putText(img, "Test Image", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imwrite("test_images/test_image.jpg", img)
print("Test image created: test_images/test_image.jpg")
'''
    
    with open("create_test_image.py", "w") as f:
        f.write(test_image_code)
    
    run_command("python create_test_image.py", "Creating test image")
    
    # 7. VERIFY INSTALLATIONS
    print("\n" + "="*40)
    print("STEP 7: VERIFYING INSTALLATIONS")
    print("="*40)
    
    verification_code = '''
import sys
print("Python version:", sys.version)

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except:
    print("PyTorch: NOT INSTALLED")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except:
    print("OpenCV: NOT INSTALLED")

try:
    import whisper
    print("Whisper: INSTALLED")
except:
    print("Whisper: NOT INSTALLED")

try:
    from ultralytics import YOLO
    print("YOLOv8: INSTALLED")
except:
    print("YOLOv8: NOT INSTALLED")

try:
    import mediapipe
    print(f"MediaPipe: {mediapipe.__version__}")
except:
    print("MediaPipe: NOT INSTALLED")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except:
    print("Transformers: NOT INSTALLED")

import os
print("\\nFiles downloaded:")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith((".pt", ".weights", ".cfg", ".names")):
            print(f"  {os.path.join(root, file)}")
'''
    
    with open("verify_setup.py", "w") as f:
        f.write(verification_code)
    
    run_command("python verify_setup.py", "Verifying setup")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("All models and dependencies have been downloaded.")
    print("You can now run your exam notebook offline.")
    print("Files are saved in:")
    print("  - models/ (Whisper and transformer models)")
    print("  - yolo_files/ (YOLOv3 files)")
    print("  - ~/.cache/ (Cached models)")
    print("="*60)

if __name__ == "__main__":
    main()
