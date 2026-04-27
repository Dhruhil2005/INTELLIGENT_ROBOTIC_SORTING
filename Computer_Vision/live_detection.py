"""
Live Camera Detection for Intelligent Robotic Sorting System
Uses trained YOLOv8 model (best.pt) for real-time object detection
"""

import cv2 # OpenCV
from ultralytics import YOLO
import numpy as np

# Configuration
MODEL_PATH = "best.pt"  # Path to your trained model
CONFIDENCE_THRESHOLD = 0.5  # Adjust based on your needs
CAMERA_INDEX = 0  # 0 for default webcam, change if using external camera

def main():
    # Load the trained model
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting live detection... Press 'q' to quit")
    
    # Main detection loop
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Run inference on the frame
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        # Draw detections on frame
        annotated_frame = results[0].plot()
        
        # Display detection information
        detections = results[0].boxes
        detection_text = f"Objects detected: {len(detections)}"
        cv2.putText(annotated_frame, detection_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Robotic Sorting - Live Detection', annotated_frame)
        
        # Print detection details to console
        if len(detections) > 0:
            print("\n--- Detected Objects ---")
            for i, box in enumerate(detections):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                # .xywh gives: [center_x, center_y, width, height]
                
                center_x, center_y, w, h = box.xywh[0].tolist()  # 1. Getting the center coordinates from the YOLO box
                # 2. Converting to integers (pixels must be whole numbers)
                ix = int(center_x)
                iy = int(center_y)
                # 3. Use these dynamic variables to 'look' at the depth map
                # This ensures you always get the depth of the MOVING object
                if 'depth_frame' in locals(): 
                    object_depth = depth_frame[iy, ix]
                    print(f"Depth: {object_depth}")
                else:
                    print("Warning: Depth frame not captured yet!")
                print(f"{i+1}. {class_name}: {confidence:.2f} found at: X={center_x:.1f}, Y={center_y:.1f}")
                
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nDetection stopped")

if __name__ == "__main__":
    main()
