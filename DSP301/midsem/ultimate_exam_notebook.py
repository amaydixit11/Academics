# DSP301 Ultimate Exam Notebook - All 5 Assignments
# Run the setup script first before internet is turned off!

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
    
    # 2. Webcam Detection Function (for demonstration - actual webcam code)
    def webcam_detection_demo():
        """Demo webcam detection code"""
        
        webcam_code = '''
import cv2
import time
from ultralytics import YOLO
import torch

# Configuration
MODEL_WEIGHTS = "yolov8n.pt"
DEVICE = "0" if torch.cuda.is_available() else "cpu"
SCORE_THRESHOLD = 0.35
WIN_NAME = "YOLOv8 Live Detection"

# Load model
model = YOLO(MODEL_WEIGHTS)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, device=DEVICE, conf=SCORE_THRESHOLD)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Add FPS text
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow(WIN_NAME, annotated_frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
'''
        
        print("Webcam Detection Code:")
        print(webcam_code)
        
        # Save the webcam code to a file
        with open("output/webcam_detection.py", "w") as f:
            f.write(webcam_code)
        print("Webcam detection code saved to output/webcam_detection.py")
    
    # 3. Object Tracking Demo
    def object_tracking_demo():
        """Demonstrate object tracking with YOLOv8"""
        
        tracking_code = '''
from ultralytics import YOLO
import cv2

# Load model for tracking
model = YOLO('yolov8n.pt')

# Start tracking
results = model.track(source=0, show=True, tracker="bytetrack.yaml")
'''
        
        print("\nObject Tracking Code:")
        print(tracking_code)
        
        # Save tracking code
        with open("output/object_tracking.py", "w") as f:
            f.write(tracking_code)
        print("Object tracking code saved to output/object_tracking.py")
    
    # 4. Video Processing Demo
    def video_processing_demo():
        """Process video file with detection"""
        
        # Create a simple test video (colored moving rectangles)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/test_video.avi', fourcc, 10.0, (640, 480))
        
        for i in range(50):  # 50 frames
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Moving rectangle
            x = int(50 + i * 10)
            y = int(200 + 50 * np.sin(i * 0.2))
            
            cv2.rectangle(frame, (x, y), (x+100, y+80), (0, 255, 0), -1)
            cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print("Test video created: output/test_video.avi")
        
        # Video processing code
        video_process_code = '''
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# Process video
results = model('test_video.avi', save=True)

# Or process and save manually
cap = cv2.VideoCapture('test_video.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('processed_video.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame)
    annotated_frame = results[0].plot()
    
    # Write frame
    out.write(annotated_frame)

cap.release()
out.release()
'''
        
        print("Video Processing Code:")
        print(video_process_code)
        
        with open("output/video_processing.py", "w") as f:
            f.write(video_process_code)
        print("Video processing code saved to output/video_processing.py")
    
    # Run demos
    webcam_detection_demo()
    object_tracking_demo()
    video_processing_demo()
    
    print("\nReal-time CV assignment complete!")

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
        
        # 1. Speech-to-Text Model (Whisper)
        stt_pipeline = pipeline("automatic-speech-recognition", 
                               model="openai/whisper-base.en", device=0)
        print("✓ Whisper Speech-to-Text model loaded")
        
        # 2. Hand Gesture Model (MediaPipe)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                              min_detection_confidence=0.7)
        mp_drawing = mp.solutions.drawing_utils
        print("✓ MediaPipe Hand Gesture model loaded")
        
        # 3. Zero-Shot Text Classification
        zero_shot_classifier = pipeline("zero-shot-classification", 
                                       model="facebook/bart-large-mnli", device=0)
        CANDIDATE_INTENTS = ["forward", "left", "right", "stop"]
        print("✓ Zero-Shot Intent NLP model loaded")
        
    except Exception as e:
        print(f"Error loading multimodal models: {e}")
        return
    
    # 2. Text Intent Classification
    def get_intent_from_text(transcript, confidence_threshold=0.60):
        """Classify text command into intent"""
        if not transcript:
            return None
        
        print(f"[NLP] Classifying text: '{transcript}'")
        
        results = zero_shot_classifier(transcript, CANDIDATE_INTENTS)
        best_intent = results['labels'][0]
        best_score = results['scores'][0]
        
        print(f"[NLP] Top classification: '{best_intent}' with confidence: {best_score:.2f}")
        
        if best_score > confidence_threshold:
            return best_intent
        else:
            print(f"[NLP] Confidence below threshold")
            return None
    
    # 3. Audio Processing
    def get_intent_from_audio(audio_path):
        """Extract intent from audio file"""
        try:
            print("\n[Audio] Transcribing speech to text...")
            transcription_result = stt_pipeline(audio_path)
            transcript = transcription_result['text'].strip().lower()
            
            return get_intent_from_text(transcript)
            
        except Exception as e:
            print(f"[Audio] Error processing audio: {e}")
            return None
    
    # 4. Video Gesture Recognition
    def get_intent_from_video(video_path):
        """Analyze video for hand gestures"""
        print("\n[Video] Analyzing video for hand gestures...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open video")
            return None
        
        gesture_counts = {"left": 0, "right": 0, "forward": 0, "stop": 0, "unknown": 0}
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 5 == 0:  # Process every 5th frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get key landmarks
                        landmarks = hand_landmarks.landmark
                        
                        # Thumb
                        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                        thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
                        
                        # Fingers
                        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                        
                        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                        
                        ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
                        ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
                        
                        pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
                        pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
                        
                        wrist = landmarks[mp_hands.HandLandmark.WRIST]
                        
                        # Gesture recognition logic
                        fingers_folded = (
                            index_tip.y > index_pip.y and
                            middle_tip.y > middle_pip.y and
                            ring_tip.y > ring_pip.y and
                            pinky_tip.y > pinky_pip.y
                        )
                        
                        fingers_open = (
                            index_tip.y < index_pip.y and
                            middle_tip.y < middle_pip.y and
                            ring_tip.y < ring_pip.y and
                            pinky_tip.y < pinky_pip.y and
                            thumb_tip.y < thumb_ip.y
                        )
                        
                        thumbs_up = (
                            thumb_tip.y < thumb_ip.y - 0.03 and
                            fingers_folded
                        )
                        
                        # Classify gesture
                        if thumbs_up:
                            gesture_counts["forward"] += 1
                        elif fingers_open:
                            gesture_counts["stop"] += 1
                        elif fingers_folded:
                            # Check thumb direction for left/right
                            if thumb_tip.x < wrist.x - 0.04:
                                gesture_counts["left"] += 1
                            elif thumb_tip.x > wrist.x + 0.04:
                                gesture_counts["right"] += 1
                            else:
                                gesture_counts["unknown"] += 1
                        else:
                            gesture_counts["unknown"] += 1
            
            frame_count += 1
        
        cap.release()
        
        # Determine dominant gesture
        if sum(gesture_counts.values()) > 0:
            dominant_gesture = max(gesture_counts, key=gesture_counts.get)
            if dominant_gesture != "unknown" and gesture_counts[dominant_gesture] > 2:
                print(f"[Video] Detected gesture counts: {gesture_counts}")
                print(f"[Video] Detected intent: '{dominant_gesture}'")
                return dominant_gesture
        
        print("[Video] No definitive gesture detected.")
        return None
    
    # 5. Multimodal Fusion
    def process_multimodal_command(video_path):
        """Main pipeline function with multimodal fusion"""
        print(f"\n{'='*20} PROCESSING MULTIMODAL COMMAND {'='*20}")
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
        
        # Extract audio from video
        temp_audio_path = "temp_audio/extracted_audio.wav"
        try:
            with VideoFileClip(video_path) as video_clip:
                if video_clip.audio:
                    video_clip.audio.write_audiofile(temp_audio_path, logger=None)
                    audio_intent = get_intent_from_audio(temp_audio_path)
                else:
                    audio_intent = None
                    print("[Audio] No audio track found in video")
        except Exception as e:
            print(f"[Audio] Error extracting audio: {e}")
            audio_intent = None
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        
        # Process video for gestures
        video_intent = get_intent_from_video(video_path)
        
        # Fusion logic
        print("\n[Fusion] Comparing intents...")
        print(f"[Fusion] Audio Intent: {audio_intent} | Video Intent: {video_intent}")
        
        if audio_intent and video_intent and audio_intent == video_intent:
            print(f"\n✓ HIGH CONFIDENCE: Intents match! Command: {audio_intent.upper()}")
            return audio_intent
        elif audio_intent and video_intent and audio_intent != video_intent:
            print(f"\n⚠ CONFLICT: Audio='{audio_intent}' vs Video='{video_intent}'. No action.")
            return None
        elif audio_intent and not video_intent:
            print(f"\n→ AUDIO ONLY: Command: {audio_intent.upper()}")
            return audio_intent
        elif video_intent and not audio_intent:
            print(f"\n→ VIDEO ONLY: Command: {video_intent.upper()}")
            return video_intent
        else:
            print("\n✗ FAILED: No clear intent detected")
            return None
    
    # 6. Create Test Content
    print("\n2. Creating Test Multimodal Content")
    
    # Create a simple test video with colored rectangles (simulating gestures)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Test videos for different gestures
    test_scenarios = {
        "forward": (0, 255, 0),  # Green for forward/thumbs up
        "stop": (0, 0, 255),     # Red for stop/open palm
        "left": (255, 0, 0),     # Blue for left
        "right": (255, 255, 0)   # Yellow for right
    }
    
    for gesture, color in test_scenarios.items():
        video_path = f"output/test_{gesture}.avi"
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
        
        for i in range(30):  # 3 seconds at 10fps
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Draw gesture representation
            if gesture == "forward":
                # Thumbs up shape
                cv2.circle(frame, (320, 240), 50, color, -1)
                cv2.rectangle(frame, (295, 290), (345, 350), color, -1)
            elif gesture == "stop":
                # Open palm shape
                cv2.circle(frame, (320, 240), 80, color, -1)
                for j in range(5):
                    cv2.rectangle(frame, (280 + j*20, 160), (295 + j*20, 220), color, -1)
            elif gesture == "left":
                # Arrow pointing left
                cv2.arrowedLine(frame, (400, 240), (240, 240), color, 10)
            else:  # right
                # Arrow pointing right
                cv2.arrowedLine(frame, (240, 240), (400, 240), color, 10)
            
            cv2.putText(frame, f"{gesture.upper()}", (250, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            out.write(frame)
        
        out.release()
        print(f"Created test video: {video_path}")
    
    # 7. Test the System
    print("\n3. Testing Multimodal System")
    
    for gesture in test_scenarios.keys():
        video_path = f"output/test_{gesture}.avi"
        if os.path.exists(video_path):
            result = process_multimodal_command(video_path)
            print(f"Test result for {gesture}: {result}")
            print("-" * 50)
    
    # 8. Complete Multimodal Code Template
    multimodal_template = '''
# Complete Multimodal AI System Template
import cv2
import mediapipe as mp
from moviepy.editor import VideoFileClip
from transformers import pipeline

class MultimodalAI:
    def __init__(self):
        # Initialize models
        self.stt_pipeline = pipeline("automatic-speech-recognition", 
                                    model="openai/whisper-base.en")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        self.classifier = pipeline("zero-shot-classification", 
                                  model="facebook/bart-large-mnli")
        self.intents = ["forward", "left", "right", "stop"]
    
    def process_text(self, text):
        results = self.classifier(text, self.intents)
        return results['labels'][0] if results['scores'][0] > 0.6 else None
    
    def process_audio(self, audio_path):
        result = self.stt_pipeline(audio_path)
        return self.process_text(result['text'])
    
    def process_video(self, video_path):
        # Video gesture recognition logic here
        pass
    
    def process_multimodal(self, video_path):
        # Complete multimodal processing
        pass

# Usage
ai = MultimodalAI()
result = ai.process_multimodal("input_video.mp4")
'''
    
    with open("output/multimodal_ai_template.py", "w") as f:
        f.write(multimodal_template)
    
    print("\nMultimodal AI assignment complete!")
    print("Template saved to: output/multimodal_ai_template.py")

# Run Assignment 5
assignment_5_multimodal()

# ============================================================================
# BONUS: EXAM HELPER FUNCTIONS
# ============================================================================

print("\n" + "="*50)
print("BONUS: EXAM HELPER FUNCTIONS")
print("="*50)

def exam_helpers():
    """Additional helper functions for exam"""
    
    print("\n1. Quick Model Loading Functions")
    
    # Quick load functions
    def quick_load_yolo():
        """Quick YOLO model loading"""
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            return model
        except:
            return None
    
    def quick_load_whisper():
        """Quick Whisper model loading"""
        try:
            import whisper
            model = whisper.load_model("base")
            return model
        except:
            return None
    
    def quick_load_transformers():
        """Quick transformer models loading"""
        try:
            from transformers import pipeline
            classifier = pipeline('sentiment-analysis')
            return classifier
        except:
            return None
    
    # Save helper functions
    helper_functions = '''
# EXAM HELPER FUNCTIONS - Copy these for quick use

def quick_image_processing(image_path):
    """Quick image processing pipeline"""
    import cv2
    import numpy as np
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(gray, 50, 150)
    
    return img, gray, blur, edges

def quick_audio_transcription(audio_path):
    """Quick audio transcription"""
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def quick_object_detection(image_path):
    """Quick object detection"""
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    results = model(image_path)
    return results

def quick_text_classification(text):
    """Quick text classification"""
    from transformers import pipeline
    classifier = pipeline('sentiment-analysis')
    return classifier(text)

def create_test_image():
    """Create a test image"""
    import cv2
    import numpy as np
    
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), (255, 0, 0), -1)
    cv2.rectangle(img, (250, 150), (400, 300), (0, 255, 0), -1)
    cv2.rectangle(img, (450, 50), (550, 350), (0, 0, 255), -1)
    cv2.putText(img, "Test", (250, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite("test_image.jpg", img)
    return img

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate"""
    from jiwer import wer
    return wer(reference, hypothesis) * 100

# Common imports for exam
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import whisper
from transformers import pipeline
import librosa
import soundfile as sf
'''
    
    with open("output/exam_helpers.py", "w") as f:
        f.write(helper_functions)
    
    print("Helper functions saved to: output/exam_helpers.py")
    
    # 2. Common Error Solutions
    print("\n2. Common Error Solutions")
    
    error_solutions = '''
# COMMON ERROR SOLUTIONS

# Error: No module named 'ultralytics'
# Solution: pip install ultralytics

# Error: CUDA out of memory
# Solution: Use device="cpu" or smaller model

# Error: Audio file not found
# Solution: Check file path and format (wav, mp3, etc.)

# Error: OpenCV video capture failed
# Solution: Check camera permissions or use different index (0, 1, 2)

# Error: Whisper model download failed
# Solution: Run whisper.load_model("tiny") first

# Error: Transformers model loading failed
# Solution: Use device=-1 for CPU or check internet connection

# Quick fixes:
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# For memory issues:
torch.cuda.empty_cache()  # Clear CUDA cache

# For audio issues:
import librosa
y, sr = librosa.load("audio.wav", sr=16000)  # Force 16kHz
'''
    
    with open("output/error_solutions.txt", "w") as f:
        f.write(error_solutions)
    
    print("Error solutions saved to: output/error_solutions.txt")
    
    print("\n3. File Structure Summary")
    
    # Show what files were created
    print("\nFiles created in output/:")
    output_files = [
        "original_image.jpg",
        "image_processing_operations.png", 
        "yolo_detection.png",
        "audio_features.png",
        "webcam_detection.py",
        "object_tracking.py",
        "video_processing.py",
        "multimodal_ai_template.py",
        "exam_helpers.py",
        "error_solutions.txt"
    ]
    
    for file in output_files:
        if os.path.exists(f"output/{file}"):
            print(f"✓ {file}")
        else:
            print(f"✗ {file}")

# Run exam helpers
exam_helpers()

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
print("="*60) SECTION 0: IMPORTS AND SETUP
# ============================================================================

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
    
    # 3. YOLOv3 Detection (Alternative)
    print("\n3. YOLOv3 Detection Implementation")
    
    def load_yolo_v3():
        """Load YOLOv3 model"""
        try:
            weights_path = "yolo_files/yolov3.weights"
            cfg_path = "yolo_files/yolov3.cfg"
            names_path = "yolo_files/coco.names"
            
            # Check if files exist
            if not all(os.path.exists(path) for path in [weights_path, cfg_path, names_path]):
                print("YOLOv3 files not found. Please run the setup script first.")
                return None, None, None
            
            # Load class names
            with open(names_path, "r") as f:
                classes = [line.strip() for line in f.readlines()]
            
            # Load network
            net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            
            # Get output layer names
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            
            return net, classes, output_layers
            
        except Exception as e:
            print(f"Error loading YOLOv3: {e}")
            return None, None, None
    
    net, classes, output_layers = load_yolo_v3()
    
    if net is not None:
        # Perform detection
        height, width = img.shape[:2]
        
        # Prepare image
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        # Process detections
        boxes, confidences, class_ids = [], [], []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    center_x, center_y, w, h = detection[:4] * np.array([width, height, width, height])
                    x, y = int(center_x - w/2), int(center_y - h/2)
                    
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Draw results
        result_img = img.copy()
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title("YOLOv3 Detection Results")
        plt.axis('off')
        plt.savefig('output/yolov3_detection.png')
        plt.show()
    
    # 4. Image Segmentation Demo
    print("\n4. Image Segmentation with Thresholding")
    
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
#