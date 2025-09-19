#!/usr/bin/env python3
"""
DSP301 Exam Setup Script
Run this FIRST before internet is turned off!
This script downloads all required models and libraries.
"""

import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path

def run_command(cmd):
    """Run shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {cmd}")
        print(f"Error: {e.stderr}")
        return False

def download_file(url, filename):
    """Download file from URL"""
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("! GPU not available, using CPU")
            return False
    except:
        print("! Could not check GPU status")
        return False

def main():
    print("="*60)
    print("DSP301 EXAM SETUP SCRIPT")
    print("="*60)
    print("This will download all required models and dependencies.")
    print("Make sure you have stable internet connection!")
    print("="*60)
    
    # Create directories
    directories = ["models", "output", "temp_audio", "screenshots", "yolo_files"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("\n" + "="*30)
    print("STEP 1: INSTALLING PYTHON PACKAGES")
    print("="*30)
    
    # Essential packages
    packages = [
        "torch torchvision torchaudio",
        "ultralytics",
        "transformers",
        "opencv-python",
        "matplotlib",
        "seaborn",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "librosa",
        "soundfile",
        "moviepy",
        "mediapipe",
        "jiwer",
        "noisereduce",
        "git+https://github.com/openai/whisper.git",
        "accelerate",
        "datasets",
        "tokenizers"
    ]
    
    for package in packages:
        print(f"\nInstalling {package}...")
        success = run_command(f"pip install {package}")
        if not success:
            print(f"⚠️  Failed to install {package}")
    
    print("\n" + "="*30)
    print("STEP 2: DOWNLOADING AI MODELS")
    print("="*30)
    
    # Download and cache models
    print("\n2.1 Downloading Whisper Models...")
    whisper_models = ["tiny", "base", "small"]
    try:
        import whisper
        for model_name in whisper_models:
            try:
                print(f"Loading Whisper {model_name}...")
                model = whisper.load_model(model_name)
                print(f"✓ Whisper {model_name} cached")
            except Exception as e:
                print(f"✗ Failed to cache Whisper {model_name}: {e}")
    except Exception as e:
        print(f"✗ Whisper not available: {e}")
    
    print("\n2.2 Downloading Transformer Models...")
    transformer_models = [
        ("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english"),
        ("zero-shot-classification", "facebook/bart-large-mnli"),
        ("automatic-speech-recognition", "openai/whisper-base.en"),
        ("text-classification", "bert-base-uncased")
    ]
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModel
        for task, model_name in transformer_models:
            try:
                print(f"Loading {model_name}...")
                if task == "text-classification" and "bert" in model_name:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                else:
                    pipe = pipeline(task, model=model_name)
                print(f"✓ {model_name} cached")
            except Exception as e:
                print(f"✗ Failed to cache {model_name}: {e}")
    except Exception as e:
        print(f"✗ Transformers not available: {e}")
    
    print("\n2.3 Downloading YOLO Models...")
    yolo_models = ["yolov8n.pt", "yolov8s.pt", "yolov8n-seg.pt"]
    try:
        from ultralytics import YOLO
        for model_name in yolo_models:
            try:
                print(f"Loading {model_name}...")
                model = YOLO(model_name)
                print(f"✓ {model_name} cached")
            except Exception as e:
                print(f"✗ Failed to cache {model_name}: {e}")
    except Exception as e:
        print(f"✗ YOLO not available: {e}")
    
    print("\n2.4 Downloading YOLOv3 Files...")
    yolo_files = {
        "yolov3.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights",
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    for filename, url in yolo_files.items():
        filepath = os.path.join("yolo_files", filename)
        if not os.path.exists(filepath):
            download_file(url, filepath)
        else:
            print(f"✓ {filename} already exists")
    
    print("\n" + "="*30)
    print("STEP 3: TESTING INSTALLATIONS")
    print("="*30)
    
    # Test imports
    test_imports = [
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("ultralytics", "Ultralytics YOLO"),
        ("whisper", "Whisper"),
        ("librosa", "Librosa"),
        ("mediapipe", "MediaPipe"),
        ("moviepy.editor", "MoviePy"),
        ("jiwer", "jiwer"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn")
    ]
    
    failed_imports = []
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name}")
            failed_imports.append(name)
    
    # Check GPU
    print(f"\nGPU Status:")
    check_gpu()
    
    # Test model loading
    print(f"\n" + "="*30)
    print("STEP 4: QUICK MODEL TESTS")
    print("="*30)
    
    # Test YOLO
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✓ YOLO model loads successfully")
    except Exception as e:
        print(f"✗ YOLO test failed: {e}")
    
    # Test Whisper
    try:
        import whisper
        model = whisper.load_model("tiny")
        print("✓ Whisper model loads successfully")
    except Exception as e:
        print(f"✗ Whisper test failed: {e}")
    
    # Test Transformers
    try:
        from transformers import pipeline
        classifier = pipeline('sentiment-analysis')
        result = classifier("This is a test")
        print("✓ Transformers pipeline works successfully")
    except Exception as e:
        print(f"✗ Transformers test failed: {e}")
    
    # Test MediaPipe
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        print("✓ MediaPipe loads successfully")
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
    
    print("\n" + "="*30)
    print("STEP 5: CREATING SAMPLE FILES")
    print("="*30)
    
    # Create sample image
    try:
        import cv2
        import numpy as np
        
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)
        cv2.rectangle(img, (250, 150), (400, 300), (0, 255, 0), -1)
        cv2.rectangle(img, (450, 50), (550, 350), (0, 0, 255), -1)
        cv2.putText(img, "SAMPLE IMAGE", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite("models/sample_image.jpg", img)
        print("✓ Sample image created")
    except Exception as e:
        print(f"✗ Failed to create sample image: {e}")
    
    # Create sample audio
    try:
        import soundfile as sf
        import numpy as np
        
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.3  # A note
        sf.write("models/sample_audio.wav", audio, sample_rate)
        print("✓ Sample audio created")
    except Exception as e:
        print(f"✗ Failed to create sample audio: {e}")
    
    # Create sample video
    try:
        import cv2
        import numpy as np
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("models/sample_video.avi", fourcc, 10.0, (640, 480))
        
        for i in range(30):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            x = int(50 + i * 10)
            cv2.rectangle(frame, (x, 200), (x+60, 240), (0, 255, 0), -1)
            cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        print("✓ Sample video created")
    except Exception as e:
        print(f"✗ Failed to create sample video: {e}")
    
    print("\n" + "="*30)
    print("SETUP SUMMARY")
    print("="*30)
    
    if failed_imports:
        print("⚠️  Some packages failed to install:")
        for package in failed_imports:
            print(f"   - {package}")
        print("\nYou may need to install these manually.")
    else:
        print("✅ All essential packages installed successfully!")
    
    print(f"\n📁 Files created:")
    print(f"   - Models cached in system")
    print(f"   - YOLOv3 files in yolo_files/")
    print(f"   - Sample files in models/")
    print(f"   - Output directories created")
    
    print(f"\n🚀 SETUP COMPLETE!")
    print(f"You can now run your exam notebook offline.")
    print(f"="*60)
    
    # Create a quick verification script
    verification_code = '''
# QUICK VERIFICATION SCRIPT
# Run this to check if everything is working

def verify_setup():
    """Verify all components are working"""
    print("Verifying exam setup...")
    
    checks = []
    
    # Check YOLO
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        checks.append(("YOLO", True))
    except:
        checks.append(("YOLO", False))
    
    # Check Whisper
    try:
        import whisper
        model = whisper.load_model("tiny")
        checks.append(("Whisper", True))
    except:
        checks.append(("Whisper", False))
    
    # Check Transformers
    try:
        from transformers import pipeline
        classifier = pipeline('sentiment-analysis')
        checks.append(("Transformers", True))
    except:
        checks.append(("Transformers", False))
    
    # Check MediaPipe
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        checks.append(("MediaPipe", True))
    except:
        checks.append(("MediaPipe", False))
    
    # Check OpenCV
    try:
        import cv2
        checks.append(("OpenCV", True))
    except:
        checks.append(("OpenCV", False))
    
    print("\\nVerification Results:")
    for component, status in checks:
        status_str = "✓" if status else "✗"
        print(f"  {status_str} {component}")
    
    success_count = sum(1 for _, status in checks if status)
    print(f"\\n{success_count}/{len(checks)} components working")
    
    if success_count == len(checks):
        print("🎉 All systems ready for exam!")
    else:
        print("⚠️  Some components need attention")

if __name__ == "__main__":
    verify_setup()
'''
    
    with open("verify_setup.py", "w") as f:
        f.write(verification_code)
    print("✓ Verification script created: verify_setup.py")
    
    # Create exam checklist
    checklist = '''
# DSP301 EXAM CHECKLIST

## Before Exam (With Internet):
□ Run setup_exam.py
□ Run verify_setup.py
□ Check all models downloaded
□ Test notebook runs without errors

## During Exam (No Internet):
□ Use the complete exam notebook
□ Reference quick_test_functions.py for help
□ Use emergency_snippets.py for code
□ Check model_loading.py for loading functions

## File Structure:
□ output/ - for results
□ temp_audio/ - for audio processing
□ yolo_files/ - YOLOv3 files
□ models/ - sample files
□ verify_setup.py - quick verification

## Emergency Commands:
- YOLO: from ultralytics import YOLO; model = YOLO('yolov8n.pt')
- Whisper: import whisper; model = whisper.load_model("base")
- Transformers: from transformers import pipeline
- MediaPipe: import mediapipe as mp
- CV: import cv2
- Audio: import librosa

## Common Issues:
- GPU not available → Use device="cpu"
- Model not found → Check internet connection during setup
- Import error → Check package installation
- Memory error → Use smaller models (tiny, base)

Good luck! 🚀
'''
    
    with open("EXAM_CHECKLIST.md", "w") as f:
        f.write(checklist)
    print("✓ Exam checklist created: EXAM_CHECKLIST.md")

if __name__ == "__main__":
    main()