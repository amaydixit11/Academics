

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  #loading the model from the pretrained model yolo 8pnt
cap = cv2.VideoCapture(0)  # Initialize webcam 

if not cap.isOpened():   # adding error handle 
    print("❌ Error: can't acces your webcam.")
    exit()
class_names = model.names

while True:  # untill the keyboard interuppt our loop will continoue
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: no frames.")
        break

    
    results = model(frame)  # fitting every frame in the yolo
    # Get detections for this frame
    detections = results[0].boxes

    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        conf = float(box.conf[0])

        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        print(f"Detected {cls_name} ({conf:.2f}) at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    annotated_frame = results[0].plot()    # printing boxes in the orignal frame and overalaping them

    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

   
    if cv2.waitKey(1) & 0xFF == ord("c"): # Exit on pressing 'c'
        break

#Cleanup
cap.release()
cv2.destroyAllWindows()
