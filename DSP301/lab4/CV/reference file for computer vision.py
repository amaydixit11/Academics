import cv2
import numpy as np
import urllib.request
import os
from google.colab.patches import cv2_imshow


def download_yolo_files():
    """Download YOLOv3 model files if they don't exist."""
    
    # URLs for YOLOv3 files
    urls = {
        'yolov3.weights': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights',
        'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
    
    # Create directory for model files
    os.makedirs('yolo_files', exist_ok=True)
    
    # Download each file if it doesn't exist
    for filename, url in urls.items():
        filepath = os.path.join('yolo_files', filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                print(f"Please download manually from: {url}")
                sys.exit(1) 
        else:
            print(f"{filename} already exists")
    
    return 'yolo_files'

def load_yolo_model(model_path):
    """Load the pre-trained YOLOv3 model and class names."""
    
    # Load class names
    classes_file = os.path.join(model_path, 'coco.names')
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Load YOLOv3 model
    config_path = os.path.join(model_path, 'yolov3.cfg')
    weights_path = os.path.join(model_path, 'yolov3.weights')
    
    print("Loading YOLOv3 model...")
    net = cv2.dnn.readNet(config_path, weights_path)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    # The way to get unconnected out layers has been updated in newer OpenCV versions.
    # This is a more robust way to do it.
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    print(f"Model loaded successfully with {len(classes)} classes")
    
    return net, classes, output_layers

def preprocess_image(image_path, target_size=(416, 416)):
    """Load and preprocess the image for YOLOv3."""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    height, width = image.shape[:2]
    
    # Create a blob from the image
    # YOLOv3 expects 416x416 input with normalization
    blob = cv2.dnn.blobFromImage(image, 1/255.0, target_size, 
                                  swapRB=True, crop=False)
    
    return image, blob, (width, height)

def detect_objects(net, blob, output_layers, conf_threshold=0.5):
    """Feed the preprocessed image to the model and get predictions."""
    
    # Set input to the network
    net.setInput(blob)
    
    # Run forward pass to get output
    outputs = net.forward(output_layers)
    
    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []
    
    # Process each output
    for output in outputs:
        for detection in output:
            # First 5 values are x, y, w, h, confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter weak predictions
            if confidence > conf_threshold:
                # Object detected
                center_x = detection[0]
                center_y = detection[1]
                w = detection[2]
                h = detection[3]
                
                # Store detection info
                boxes.append([center_x, center_y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def apply_nms(boxes, confidences, class_ids, width, height, 
              conf_threshold=0.5, nms_threshold=0.4):
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    
    # Convert YOLO format to OpenCV format
    opencv_boxes = []
    for box in boxes:
        center_x, center_y, w, h = box
        
        # Convert to actual coordinates
        center_x = int(center_x * width)
        center_y = int(center_y * height)
        w = int(w * width)
        h = int(h * height)
        
        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)
        
        opencv_boxes.append([x, y, w, h])
    
    # Apply NMS
    if len(opencv_boxes) > 0:
        indices = cv2.dnn.NMSBoxes(opencv_boxes, confidences, 
                                   conf_threshold, nms_threshold)
        
        # Get final detections
        final_boxes = []
        final_confidences = []
        final_class_ids = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(opencv_boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])
        
        return final_boxes, final_confidences, final_class_ids
    
    return [], [], []

def draw_detections(image, boxes, confidences, class_ids, classes):
    """Draw bounding boxes and labels on the image."""
    
    # Generate random colors for each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
    
    # Draw each detection
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        
        # Get class name and confidence
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        
        # Get color for this class
        color = [int(c) for c in colors[class_ids[i]]]
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_y = y - 10 if y - 10 > 10 else y + 10
        
        cv2.rectangle(image, (x, label_y - label_size[1] - 4),
                     (x + label_size[0], label_y + 4), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image

def main():
    """Main function to run YOLOv3 object detection."""
    
    # Step 1: Download YOLOv3 model files
    model_path = download_yolo_files()
    
    # Step 2: Load the model and class names
    net, classes, output_layers = load_yolo_model(model_path)
    
    # Step 3: Load and preprocess the image
    # You can change this to any image path
    image_path = 'sample_image.jpg'  # Change this to your image path
    
    try:
        image, blob, (width, height) = preprocess_image(image_path)
        print(f"Image loaded: {width}x{height}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Using a sample image URL instead...")
        # Download a sample image if local image not found
        sample_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/basketball2.png'
        urllib.request.urlretrieve(sample_url, 'sample_image.jpg')
        image, blob, (width, height) = preprocess_image('sample_image.jpg')
    
    # Step 4 & 5: Perform detection
    print("Running object detection...")
    boxes, confidences, class_ids = detect_objects(net, blob, output_layers, 
                                                   conf_threshold=0.5)
    print(f"Detected {len(boxes)} objects before NMS")
    
    # Step 6: Apply Non-Maximum Suppression
    final_boxes, final_confidences, final_class_ids = apply_nms(
        boxes, confidences, class_ids, width, height,
        conf_threshold=0.5, nms_threshold=0.4
    )
    print(f"Detected {len(final_boxes)} objects after NMS")
    
    # Step 7: Draw bounding boxes and display result
    result_image = draw_detections(image, final_boxes, final_confidences, 
                                   final_class_ids, classes)
    
    # Save the result
    output_path = 'detection_result.jpg'
    cv2.imwrite(output_path, result_image)
    print(f"Result saved to {output_path}")
    
    # Display the result (if running in a GUI environment)
    print("Displaying detection result:")
    cv2_imshow(result_image)
    
    # Print detection summary
    print("\nDetection Summary:")
    for i in range(len(final_boxes)):
        class_name = classes[final_class_ids[i]]
        confidence = final_confidences[i]
        print(f"- {class_name}: {confidence:.2%}")

if __name__ == "__main__":
    main()