# DSP301 Ultimate Exam Notebook - All 5 Assignments
# Run the setup script first before internet is turned off!

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Audio, Video
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs("temp_audio", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

print("Environment setup complete!")

# ============================================================================
# ASSIGNMENT 1: TEXT PROCESSING & TRANSFORMERS
# ============================================================================

print("\n" + "="*50)
print("ASSIGNMENT 1: TEXT PROCESSING & TRANSFORMERS")
print("="*50)

def assignment_1_text_processing():
    """Complete text processing assignment with transformers"""
    
    # 1. BERT Text Classification
    print("\n1. BERT Text Classification")
    try:
        from transformers import pipeline, AutoTokenizer, AutoModel
        
        # Initialize classifier
        classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        
        # Test sentences
        test_sentences = [
            "I love this movie!",
            "This is terrible.",
            "The weather is okay today.",
            "Machine learning is fascinating!"
        ]
        
        print("Text Classification Results:")
        for sentence in test_sentences:
            result = classifier(sentence)
            print(f"Text: '{sentence}'")
            print(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.3f}")
            print("-" * 40)
            
    except Exception as e:
        print(f"BERT Classification error: {e}")
    
    # 2. Text Tokenization
    print("\n2. Text Tokenization Examples")
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Tokenization is important for NLP.",
            "This sentence has some unusual-words and punctuation!"
        ]
        
        for text in sample_texts:
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            print(f"Original: {text}")
            print(f"Tokens: {tokens}")
            print(f"Token IDs: {token_ids}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Tokenization error: {e}")
    
    # 3. Zero-Shot Classification
    print("\n3. Zero-Shot Text Classification")
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        texts = [
            "I want to book a flight to Paris",
            "Turn on the lights in the living room",
            "What's the weather like tomorrow?",
            "Play some jazz music"
        ]
        
        labels = ["travel", "smart_home", "weather", "music", "food", "sports"]
        
        for text in texts:
            result = classifier(text, labels)
            print(f"Text: '{text}'")
            print(f"Top prediction: {result['labels'][0]} ({result['scores'][0]:.3f})")
            print(f"All scores: {dict(zip(result['labels'][:3], [f'{s:.3f}' for s in result['scores'][:3]]))}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Zero-shot classification error: {e}")

# Run Assignment 1
assignment_1_text_processing()

# ============================================================================
# ASSIGNMENT 2: AUDIO PROCESSING & ASR
# ============================================================================

print("\n" + "="*50)
print("ASSIGNMENT 2: AUDIO PROCESSING & ASR")
print("="*50)

def assignment_2_audio_processing():
    """Complete audio processing and ASR assignment"""
    
    # Audio processing libraries
    try:
        import whisper
        import librosa
        import soundfile as sf
        import noisereduce as nr
        from jiwer import wer
        import scipy.io.wavfile as wav
        
        print("Audio libraries loaded successfully!")
        
    except ImportError as e:
        print(f"Missing audio library: {e}")
        return
    
    # Reference sentence for WER calculation
    REFERENCE_SENTENCE = "The quick brown fox jumps over the lazy dog"
    
    # 1. Load Whisper Model
    print("\n1. Loading Whisper ASR Model")
    try:
        model = whisper.load_model("base")
        print("Whisper model loaded successfully!")
    except Exception as e:
        print(f"Error loading Whisper: {e}")
        return
    
    # 2. Noise Reduction Function
    def apply_noise_reduction(audio_path, output_path):
        """Apply noise reduction to audio file"""
        try:
            # Read audio file
            rate, data = wav.read(audio_path)
            
            # Convert to float if needed
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            
            # Apply simple threshold-based noise reduction
            threshold = 0.01
            mask = np.abs(data) > threshold
            cleaned_data = data * mask
            
            # Normalize
            cleaned_data = cleaned_data / np.max(np.abs(cleaned_data)) * 0.8
            
            # Convert back to int16
            cleaned_data = (cleaned_data * 32767).astype(np.int16)
            
            # Save cleaned audio
            wav.write(output_path, rate, cleaned_data)
            print(f"Noise reduction applied. Saved: {output_path}")
            
        except Exception as e:
            print(f"Noise reduction error: {e}")
    
    # 3. Transcription Function
    def transcribe_audio(audio_path, label="Audio"):
        """Transcribe audio using Whisper"""
        try:
            if not os.path.exists(audio_path):
                return f"FILE NOT FOUND: {audio_path}"
            
            result = model.transcribe(audio_path)
            transcription = result["text"].strip()
            print(f"{label} transcription: '{transcription}'")
            return transcription
            
        except Exception as e:
            print(f"Transcription error for {label}: {e}")
            return "ERROR"
    
    # 4. WER Calculation Function
    def calculate_wer(reference, hypothesis, label=""):
        """Calculate Word Error Rate"""
        try:
            if hypothesis.startswith(("FILE NOT FOUND", "ERROR")):
                print(f"{label} WER: Cannot calculate (transcription failed)")
                return None
            
            error_rate = wer(reference.lower(), hypothesis.lower())
            percentage = error_rate * 100
            print(f"{label} WER: {percentage:.1f}%")
            return percentage
            
        except Exception as e:
            print(f"WER calculation error for {label}: {e}")
            return None
    
    # 5. Create Sample Audio (for demonstration)
    print("\n2. Creating Sample Audio Files")
    
    # Create a simple sine wave as test audio
    sample_rate = 16000
    duration = 2.0
    frequency = 440  # A note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    clean_audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Add noise to create "noisy" version
    noise = np.random.normal(0, 0.1, len(clean_audio))
    noisy_audio = clean_audio + noise
    
    # Save sample files
    clean_path = "temp_audio/clean_sample.wav"
    noisy_path = "temp_audio/noisy_sample.wav"
    
    sf.write(clean_path, clean_audio, sample_rate)
    sf.write(noisy_path, noisy_audio, sample_rate)
    
    print("Sample audio files created!")
    
    # 6. Demonstrate the pipeline
    print("\n3. Audio Processing Pipeline Demonstration")
    
    # Apply noise reduction
    cleaned_path = "temp_audio/cleaned_sample.wav"
    apply_noise_reduction(noisy_path, cleaned_path)
    
    # Transcribe all versions (note: sine waves won't produce meaningful text)
    print("\n4. Transcription Results (Note: Sine waves won't produce meaningful speech)")
    transcribe_audio(clean_path, "Clean Audio")
    transcribe_audio(noisy_path, "Noisy Audio")
    transcribe_audio(cleaned_path, "Cleaned Audio")
    
    # 7. Audio Feature Extraction Demo
    print("\n5. Audio Feature Extraction")
    
    try:
        # Load audio with librosa
        y, sr = librosa.load(clean_path)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        print(f"Audio duration: {len(y)/sr:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        print(f"MFCC shape: {mfccs.shape}")
        print(f"Spectral centroid mean: {np.mean(spectral_centroid):.2f}")
        print(f"Zero crossing rate mean: {np.mean(zero_crossing_rate):.4f}")
        
        # Plot features
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform')
        
        plt.subplot(3, 1, 2)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title('MFCCs')
        
        plt.subplot(3, 1, 3)
        times = librosa.times_like(spectral_centroid)
        plt.plot(times, spectral_centroid[0])
        plt.title('Spectral Centroid')
        plt.xlabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig('output/audio_features.png')
        plt.show()
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
    
    print("Audio processing assignment complete!")

# Run Assignment 2
assignment_2_audio_processing()

# ============================================================================
# ASSIGNMENT 3: COMPUTER VISION & OBJECT DETECTION
# ============================================================================

print("\n" + "="*50)
print("ASSIGNMENT 3: COMPUTER VISION & OBJECT DETECTION")
print("="*50)

def assignment_3_computer_vision():
    """Complete computer vision assignment"""
    
    # 1. Basic Image Processing
    print("\n1. Basic Image Processing Operations")
    
    # Create a sample image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)  # Blue
    cv2.rectangle(img, (250, 150), (400, 300), (0, 255, 0), -1)  # Green
    cv2.rectangle(img, (450, 50), (550, 350), (0, 0, 255), -1)  # Red
    cv2.putText(img, "Sample Image", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save original
    cv2.imwrite("output/original_image.jpg", img)
    
    # Basic operations
    operations = {
        "Grayscale": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "Blur": cv2.GaussianBlur(img, (15, 15), 0),
        "Edge Detection": cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150),
        "Histogram Equalization": cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    }
    
    # Display results
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    for i, (name, processed_img) in enumerate(operations.items(), 2):
        plt.subplot(3, 2, i)
        if len(processed_img.shape) == 3:
            plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(processed_img, cmap='gray')
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/image_processing_operations.png')
    plt.show()
    
    # 2. YOLOv8 Object Detection
    print("\n2. YOLOv8 Object Detection")
    
    try:
        from ultralytics import YOLO
        
        # Load YOLOv8 model
        model = YOLO('yolov8n.pt')
        
        # Run detection on sample image
        results = model(img)
        
        # Plot results
        annotated_img = results[0].plot()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.title("YOLOv8 Detection Results")
        plt.axis('off')
        plt.savefig('output/yolo_detection.png')
        plt.show()
        
        # Print detection results
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"Detected: {class_name} (confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"YOLOv8 error: {e}")
    
    # 3. Image Segmentation Demo
    print("\n3. Image Segmentation with Thresholding")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Different thresholding methods
    thresholding_methods = {
        "Binary": cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
        "Binary Inverse": cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1],
        "Otsu's": cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        "Adaptive Mean": cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
        "Adaptive Gaussian": cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, thresh_img) in enumerate(thresholding_methods.items(), 1):
        plt.subplot(2, 3, i)
        plt.imshow(thresh_img, cmap='gray')
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/thresholding_methods.png')
    plt.show()
    
    print("Computer vision assignment complete!")

# Run Assignment 3
assignment_3_computer_vision()

# ============================================================================
# ASSIGNMENT 4: REAL-TIME WEBCAM COMPUTER VISION
# ============================================================================

print("\n" + "="*50)
print("ASSIGNMENT 4: REAL-TIME WEBCAM COMPUTER VISION")
print("="*50)

def assignment_4_realtime_cv():
    """Real-time webcam computer vision with YOLOv8"""
    
    print("\n1. Real-time Detection and Tracking Setup")
    
    try:
        from ultralytics import YOLO
        import torch
        
        # Configuration
        MODEL_WEIGHTS = "yolov8n.pt" 
        DEVICE = "0" if torch.cuda.is_available() else "cpu"
        SCORE_THRESHOLD = 0.35
        
        print(f"Device: {DEVICE}")
        print(f"Model: {MODEL_WEIGHTS}")
        print(f"Confidence threshold: {SCORE_THRESHOLD}")
        
        # Load model
        model = YOLO(MODEL_WEIGHTS)
        print("YOLOv8 model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading YOLOv8: {e}")
        return
    
    # 2. Live Webcam Detection Code Template
    print("\n2. Live Webcam Detection Code")
    
    webcam_code = '''
import cv2
import time
from ultralytics import YOLO
import torch
import numpy as np
import os

# Configuration
MODEL_WEIGHTS = "yolov8n.pt"
DEVICE = "0" if torch.cuda.is_available() else "cpu"
SCORE_THRESHOLD = 0.35
WIN_NAME = "YOLOv8 Live Detection & Tracking"
OUT_DIR = "screenshots"
VIDEO_OUT = "live_detection.avi"
FPS_RECORD = 20.0

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_WEIGHTS)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(VIDEO_OUT, fourcc, FPS_RECORD, (frame_width, frame_height))

# Create named window
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

# Use model.track() for tracking
stream = model.track(source=0, stream=True, device=DEVICE)

frame_count = 0
start_time = time.time()

try:
    for result in stream:
        # Get original frame
        frame = result.orig_img
        if frame is None:
            continue
            
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Get detections with tracking
        boxes = result.boxes
        if boxes is not None:
            # Draw detections manually for more control
            for i, box in enumerate(boxes):
                if box.conf[0] < SCORE_THRESHOLD:
                    continue
                    
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                # Get tracking ID if available
                track_id = int(box.id[0]) if box.id is not None else None
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create label with tracking info
                id_text = f"ID:{track_id}" if track_id else "ID:-"
                label = f"{cls_name} {int(conf*100)}% {id_text}"
                
                # Draw label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow(WIN_NAME, frame)
        
        # Save frame to video
        video_writer.write(frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = int(time.time())
            filename = os.path.join(OUT_DIR, f"screenshot_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video saved: {VIDEO_OUT}")
'''
    
    # Save the webcam code
    with open("output/live_webcam_detection.py", "w") as f:
        f.write(webcam_code)
    print("Live webcam code saved to: output/live_webcam_detection.py")
    
    # 3. Video Processing Demo
    print("\n3. Video File Processing")
    
    # Create test video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    test_video_path = 'output/test_moving_objects.avi'
    out = cv2.VideoWriter(test_video_path, fourcc, 10.0, (640, 480))
    
    for i in range(60):  # 6 seconds at 10fps
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Moving rectangle (simulating a car)
        x = int(50 + i * 8)
        y = 200
        cv2.rectangle(frame, (x, y), (x+80, y+40), (0, 255, 0), -1)
        
        # Moving circle (simulating a ball)
        cx = int(320 + 100 * np.sin(i * 0.3))
        cy = int(100 + 30 * np.cos(i * 0.2))
        cv2.circle(frame, (cx, cy), 25, (255, 0, 0), -1)
        
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    print(f"Test video created: {test_video_path}")
    
    # Video processing code
    video_process_code = '''
from ultralytics import YOLO
import cv2

def process_video(input_path, output_path):
    """Process video file with YOLO detection"""
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Process video (automatic save)
    results = model(input_path, save=True, project="runs/detect", name="video_results")
    
    # Manual processing for custom output
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        
        # Add frame number
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        out.write(annotated_frame)
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"Processed video saved: {output_path}")

# Usage
process_video("test_moving_objects.avi", "processed_output.avi")
'''
    
    with open("output/video_processing_complete.py", "w") as f:
        f.write(video_process_code)
    print("Video processing code saved to: output/video_processing_complete.py")
    
    print("Real-time CV assignment complete!")

# Run Assignment 4
assignment_4_realtime_cv()

# ============================================================================
# ASSIGNMENT 5: MULTIMODAL AI (AUDIO + VIDEO + TEXT)
# ============================================================================

print("\n" + "="*50)
print("ASSIGNMENT 5: MULTIMODAL AI SYSTEM")
print("="*50)

def assignment_5_multimodal():
    """Complete multimodal AI assignment"""
    
    print("\n1. Loading Multimodal Models")
    
    try:
        import whisper
        import mediapipe as mp
        from transformers import pipeline
        from moviepy.editor import VideoFileClip
        
        # 1. Speech-to-Text Model
        print("Loading Whisper for speech recognition...")
        try:
            whisper_model = whisper.load_model("base")
            print("✓ Whisper model loaded")
        except:
            stt_pipeline = pipeline("automatic-speech-recognition", 
                                   model="openai/whisper-base.en")
            print("✓ Whisper pipeline loaded")
        
        # 2. Hand Gesture Model
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                              min_detection_confidence=0.7)
        mp_drawing = mp.solutions.drawing_utils
        print("✓ MediaPipe Hand model loaded")
        
        # 3. Zero-Shot Text Classification
        zero_shot_classifier = pipeline("zero-shot-classification", 
                                       model="facebook/bart-large-mnli")
        CANDIDATE_INTENTS = ["forward", "left", "right", "stop"]
        print("✓ Zero-Shot classifier loaded")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # 2. Complete Multimodal System
    multimodal_system_code = '''
import cv2
import os
import mediapipe as mp
from moviepy.editor import VideoFileClip
from transformers import pipeline
import whisper
import numpy as np

class MultimodalCommandSystem:
    def __init__(self):
        """Initialize all AI models for multimodal processing"""
        print("Initializing Multimodal AI System...")
        
        # Speech-to-Text
        try:
            self.whisper_model = whisper.load_model("base")
            self.stt_method = "whisper"
        except:
            self.stt_pipeline = pipeline("automatic-speech-recognition", 
                                        model="openai/whisper-base.en")
            self.stt_method = "pipeline"
        
        # Hand Gesture Recognition
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Text Intent Classification
        self.intent_classifier = pipeline("zero-shot-classification", 
                                         model="facebook/bart-large-mnli")
        self.candidate_intents = ["forward", "left", "right", "stop"]
        
        print("✓ All models loaded successfully!")
    
    def extract_audio_from_video(self, video_path, audio_path):
        """Extract audio from video file"""
        try:
            with VideoFileClip(video_path) as video_clip:
                if video_clip.audio:
                    video_clip.audio.write_audiofile(audio_path, logger=None, verbose=False)
                    return True
                else:
                    print("[Audio] No audio track found in video")
                    return False
        except Exception as e:
            print(f"[Audio] Error extracting audio: {e}")
            return False
    
    def transcribe_audio(self, audio_path):
        """Convert speech to text"""
        try:
            if self.stt_method == "whisper":
                result = self.whisper_model.transcribe(audio_path)
                return result["text"].strip().lower()
            else:
                result = self.stt_pipeline(audio_path)
                return result['text'].strip().lower()
        except Exception as e:
            print(f"[Audio] Transcription error: {e}")
            return None
    
    def classify_text_intent(self, text, confidence_threshold=0.6):
        """Classify text into command intent"""
        if not text:
            return None, 0.0
        
        try:
            results = self.intent_classifier(text, self.candidate_intents)
            best_intent = results['labels'][0]
            best_score = results['scores'][0]
            
            print(f"[NLP] Text: '{text}' -> Intent: {best_intent} ({best_score:.3f})")
            
            if best_score > confidence_threshold:
                return best_intent, best_score
            else:
                return None, best_score
                
        except Exception as e:
            print(f"[NLP] Classification error: {e}")
            return None, 0.0
    
    def analyze_hand_gestures(self, video_path):
        """Analyze video for hand gesture commands"""
        print("[Video] Analyzing hand gestures...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        gesture_counts = {"left": 0, "right": 0, "forward": 0, "stop": 0}
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 3 == 0:  # Process every 3rd frame for speed
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        gesture = self.classify_gesture(hand_landmarks.landmark)
                        if gesture:
                            gesture_counts[gesture] += 1
            
            frame_count += 1
        
        cap.release()
        
        # Find dominant gesture
        if sum(gesture_counts.values()) > 0:
            dominant = max(gesture_counts, key=gesture_counts.get)
            if gesture_counts[dominant] >= 3:  # Minimum confidence
                print(f"[Video] Gesture counts: {gesture_counts}")
                print(f"[Video] Dominant gesture: {dominant}")
                return dominant
        
        print("[Video] No clear gesture detected")
        return None
    
    def classify_gesture(self, landmarks):
        """Classify hand landmarks into gesture commands"""
        # Get key landmarks
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = landmarks[self.mp_hands.HandLandmark.PINKY_PIP]
        
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # Gesture recognition logic
        fingers_extended = (
            index_tip.y < index_pip.y and
            middle_tip.y < middle_pip.y and
            ring_tip.y < ring_pip.y and
            pinky_tip.y < pinky_pip.y
        )
        
        thumb_up = thumb_tip.y < thumb_ip.y - 0.02
        fingers_closed = not fingers_extended
        
        # Classify gestures
        if thumb_up and fingers_closed:
            return "forward"  # Thumbs up
        elif fingers_extended and thumb_up:
            return "stop"     # Open palm
        elif fingers_closed:
            # Check thumb direction for left/right
            if thumb_tip.x < wrist.x - 0.05:
                return "left"
            elif thumb_tip.x > wrist.x + 0.05:
                return "right"
        
        return None
    
    def process_multimodal_command(self, video_path):
        """Main function to process multimodal input"""
        print(f"\\n{'='*50}")
        print(f"PROCESSING MULTIMODAL COMMAND: {os.path.basename(video_path)}")
        print(f"{'='*50}")
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return None
        
        # Process audio
        temp_audio = "temp_audio_extract.wav"
        audio_intent = None
        
        if self.extract_audio_from_video(video_path, temp_audio):
            transcript = self.transcribe_audio(temp_audio)
            if transcript:
                audio_intent, audio_conf = self.classify_text_intent(transcript)
            os.remove(temp_audio)  # Cleanup
        
        # Process video gestures
        video_intent = self.analyze_hand_gestures(video_path)
        
        # Multimodal fusion decision
        print(f"\\n[FUSION] Audio Intent: {audio_intent}")
        print(f"[FUSION] Video Intent: {video_intent}")
        
        if audio_intent and video_intent:
            if audio_intent == video_intent:
                print(f"\\n✅ HIGH CONFIDENCE MATCH: {audio_intent.upper()}")
                return audio_intent
            else:
                print(f"\\n⚠️  CONFLICT: Audio={audio_intent} vs Video={video_intent}")
                return "conflict"
        elif audio_intent:
            print(f"\\n🎤 AUDIO COMMAND: {audio_intent.upper()}")
            return audio_intent
        elif video_intent:
            print(f"\\n👋 GESTURE COMMAND: {video_intent.upper()}")
            return video_intent
        else:
            print(f"\\n❌ NO COMMAND DETECTED")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize system
    system = MultimodalCommandSystem()
    
    # Process a video file
    result = system.process_multimodal_command("test_command_video.mp4")
    
    if result and result != "conflict":
        print(f"\\n🤖 EXECUTING COMMAND: {result.upper()}")
        # Add your robot control code here
    else:
        print("\\n🛑 NO ACTION TAKEN")
'''
    
    # Save the complete system
    with open("output/multimodal_ai_complete.py", "w") as f:
        f.write(multimodal_system_code)
    print("Complete multimodal system saved to: output/multimodal_ai_complete.py")
    
    # 3. Create Test Videos
    print("\n2. Creating Test Videos for Different Commands")
    
    test_commands = {
        "forward": {"color": (0, 255, 0), "shape": "thumbs_up"},
        "stop": {"color": (0, 0, 255), "shape": "open_palm"}, 
        "left": {"color": (255, 0, 0), "shape": "arrow_left"},
        "right": {"color": (255, 255, 0), "shape": "arrow_right"}
    }
    
    for command, props in test_commands.items():
        video_path = f"output/test_multimodal_{command}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
        
        for i in range(40):  # 4 seconds
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            color = props["color"]
            
            # Draw different shapes for different commands
            if props["shape"] == "thumbs_up":
                cv2.circle(frame, (320, 200), 40, color, -1)
                cv2.rectangle(frame, (300, 240), (340, 320), color, -1)
            elif props["shape"] == "open_palm":
                cv2.circle(frame, (320, 240), 60, color, -1)
                for j in range(5):
                    cv2.rectangle(frame, (280 + j*16, 180), (290 + j*16, 220), color, -1)
            elif props["shape"] == "arrow_left":
                pts = np.array([[400, 240], [280, 200], [280, 220], [320, 240], 
                               [280, 260], [280, 280]], np.int32)
                cv2.fillPoly(frame, [pts], color)
            else:  # arrow_right
                pts = np.array([[240, 240], [360, 200], [360, 220], [320, 240], 
                               [360, 260], [360, 280]], np.int32)
                cv2.fillPoly(frame, [pts], color)
            
            # Add command text
            cv2.putText(frame, command.upper(), (220, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(frame, f"Frame {i+1}/40", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"Created test video: {video_path}")
    
    print("\nMultimodal AI assignment complete!")

# Run Assignment 5
assignment_5_multimodal()

# ============================================================================
# EXAM HELPER FUNCTIONS AND UTILITIES
# ============================================================================

print("\n" + "="*50)
print("EXAM HELPER FUNCTIONS")
print("="*50)

def create_exam_helpers():
    """Create comprehensive exam helper functions"""
    
    # 1. Quick Test Functions
    print("\n1. Creating Quick Test Functions")
    
    quick_test_code = '''
# QUICK TEST FUNCTIONS FOR EXAM

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_test_image(filename="test_image.jpg"):
    """Create a test image with colored shapes"""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Draw different colored rectangles
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)    # Blue
    cv2.rectangle(img, (250, 150), (400, 300), (0, 255, 0), -1)  # Green  
    cv2.rectangle(img, (450, 50), (550, 350), (0, 0, 255), -1)   # Red
    
    # Add text
    cv2.putText(img, "TEST IMAGE", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(filename, img)
    print(f"Test image created: {filename}")
    return img

def create_test_audio(filename="test_audio.wav", duration=3.0):
    """Create a test audio file (sine wave)"""
    import soundfile as sf
    
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A note
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    sf.write(filename, audio, sample_rate)
    print(f"Test audio created: {filename}")
    return audio

def create_test_video(filename="test_video.avi", duration_frames=50):
    """Create a test video with moving objects"""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 10.0, (640, 480))
    
    for i in range(duration_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Moving rectangle
        x = int(50 + i * 8)
        y = 200
        cv2.rectangle(frame, (x, y), (x+60, y+40), (0, 255, 0), -1)
        
        # Frame counter
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    print(f"Test video created: {filename}")

def quick_yolo_detection(image_path):
    """Quick YOLO detection on image"""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        results = model(image_path)
        
        # Print detections
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    print(f"Detected: {cls_name} (confidence: {conf:.2f})")
        
        return results
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return None

def quick_whisper_transcription(audio_path):
    """Quick audio transcription with Whisper"""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        print(f"Transcription: {result['text']}")
        return result['text']
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return None

def quick_text_classification(text, labels=None):
    """Quick text classification"""
    try:
        from transformers import pipeline
        
        if labels:
            # Zero-shot classification
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            result = classifier(text, labels)
            print(f"Text: '{text}'")
            print(f"Top classification: {result['labels'][0]} ({result['scores'][0]:.3f})")
            return result
        else:
            # Sentiment analysis
            classifier = pipeline('sentiment-analysis')
            result = classifier(text)
            print(f"Text: '{text}'")
            print(f"Sentiment: {result[0]['label']} ({result[0]['score']:.3f})")
            return result
            
    except Exception as e:
        print(f"Text classification error: {e}")
        return None

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate"""
    try:
        from jiwer import wer
        error_rate = wer(reference.lower(), hypothesis.lower()) * 100
        print(f"Reference: '{reference}'")
        print(f"Hypothesis: '{hypothesis}'")
        print(f"WER: {error_rate:.1f}%")
        return error_rate
    except Exception as e:
        print(f"WER calculation error: {e}")
        return None

# Usage examples:
if __name__ == "__main__":
    # Create test files
    create_test_image()
    create_test_audio()
    create_test_video()
    
    # Test YOLO
    quick_yolo_detection("test_image.jpg")
    
    # Test Whisper (won't work well with sine wave)
    quick_whisper_transcription("test_audio.wav")
    
    # Test text classification
    quick_text_classification("I love machine learning!", ["positive", "negative", "neutral"])
    
    # Test WER
    calculate_wer("The quick brown fox", "The quick brown fox")
'''
    
    with open("output/quick_test_functions.py", "w") as f:
        f.write(quick_test_code)
    print("Quick test functions saved to: output/quick_test_functions.py")
    
    # 2. Emergency Code Snippets
    print("\n2. Creating Emergency Code Snippets")
    
    emergency_code = '''
# EMERGENCY CODE SNIPPETS - COPY AND PASTE READY

# Import essentials
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directories
os.makedirs("output", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# YOLO Quick Setup
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('image.jpg')
results[0].plot()

# Whisper Quick Setup
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.wav")
print(result["text"])

# MediaPipe Hands Quick Setup
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Transformers Quick Setup
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
result = classifier("I love this!")

# Zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier("Turn left", ["left", "right", "forward", "stop"])

# Image processing basics
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (15, 15), 0)
edges = cv2.Canny(gray, 50, 150)

# Audio processing basics
import librosa
y, sr = librosa.load('audio.wav')
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# WER calculation
from jiwer import wer
error_rate = wer("reference text", "hypothesis text")
print(f"WER: {error_rate * 100:.1f}%")

# Video processing
cap = cv2.VideoCapture('video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Process frame here
    out.write(frame)

cap.release()
out.release()

# Webcam capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''
    
    with open("output/emergency_snippets.py", "w") as f:
        f.write(emergency_code)
    print("Emergency code snippets saved to: output/emergency_snippets.py")
    
    # 3. Model Loading Functions
    print("\n3. Creating Model Loading Functions")
    
    model_loading_code = '''
# MODEL LOADING FUNCTIONS

def load_yolo(model_size='n'):
    """Load YOLO model"""
    try:
        from ultralytics import YOLO
        model = YOLO(f'yolov8{model_size}.pt')
        print(f"YOLO v8{model_size} loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading YOLO: {e}")
        return None

def load_whisper(size='base'):
    """Load Whisper model"""
    try:
        import whisper
        model = whisper.load_model(size)
        print(f"Whisper {size} loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading Whisper: {e}")
        return None

def load_transformers_pipeline(task, model=None):
    """Load transformers pipeline"""
    try:
        from transformers import pipeline
        if model:
            pipe = pipeline(task, model=model)
        else:
            pipe = pipeline(task)
        print(f"Transformers {task} pipeline loaded")
        return pipe
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None

def load_mediapipe_hands():
    """Load MediaPipe hands"""
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        print("MediaPipe hands loaded successfully")
        return hands, mp_hands
    except Exception as e:
        print(f"Error loading MediaPipe: {e}")
        return None, None

# Quick test all models
if __name__ == "__main__":
    yolo = load_yolo()
    whisper_model = load_whisper()
    sentiment = load_transformers_pipeline('sentiment-analysis')
    zero_shot = load_transformers_pipeline('zero-shot-classification', 'facebook/bart-large-mnli')
    hands, mp_hands = load_mediapipe_hands()
'''
    
    with open("output/model_loading.py", "w") as f:
        f.write(model_loading_code)
    print("Model loading functions saved to: output/model_loading.py")

# Run helper creation
create_exam_helpers()

print("\n" + "="*60)
print("🎉 ALL ASSIGNMENTS COMPLETE!")
print("="*60)
print("This notebook covers all 5 assignments:")
print("1. ✓ Text Processing & Transformers")
print("2. ✓ Audio Processing & ASR") 
print("3. ✓ Computer Vision & Object Detection")
print("4. ✓ Real-time Webcam CV")
print("5. ✓ Multimodal AI System")
print("\nBonus: ✓ Exam Helper Functions")
print("\nRun the setup script BEFORE the exam!")
print("Good luck! 🚀")
print("="*60)