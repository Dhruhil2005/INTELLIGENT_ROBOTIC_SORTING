#!/usr/bin/env python3
"""
Computer Vision System for Waste Detection
Detects: Plastic Bottles, Aluminum Cans, Juice Boxes

Author: Dhruhil Gajera
Student ID: 240108542
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
from collections import deque
import os
import json

class WasteDetectionCV:
    """Computer Vision System for Waste Sorting"""
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize CV system
        
        Args:
            model_path: Path to YOLO model ('yolov8n.pt' for pretrained)
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
        """
        print("="*70)
        print("WASTE DETECTION - COMPUTER VISION SYSTEM")
        print("Student ID: 240108542 | Aston University")
        print("="*70)
        
        print("\n[INITIALIZING]")
        print(f"Loading model: {model_path}")
        
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Class names mapping (for pretrained COCO model)
        # COCO class 39 = bottle
        self.target_classes = {
            39: 'plastic_bottle',  # COCO bottle class
            # Add more as needed
        }
        
        # Colors for visualization (BGR format)
        self.colors = {
            'plastic_bottle': (0, 255, 0),     # Green
            'aluminum_can': (255, 0, 0),       # Blue  
            'juice_box': (0, 165, 255),        # Orange
            'unknown': (128, 128, 128)         # Gray
        }
        
        # Performance metrics
        self.fps_history = deque(maxlen=250)
        self.detection_log = []
        self.frame_count = 0
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'plastic_bottles': 0,
            'aluminum_cans': 0,
            'juice_boxes': 0,
            'avg_confidence': 0.0
        }
        
        print("✓ Model loaded successfully")
        print(f"✓ Confidence threshold: {confidence_threshold}")
        print()
    
    def detect_objects(self, frame):
        """
        Detect waste objects in frame
        
        Returns:
            List of detections with bbox, class, confidence, center position
        """
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # Extract box information
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Map to waste categories
                waste_type = self._classify_waste(class_id, class_name)
                
                if waste_type:  # Only include relevant objects
                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (center_x, center_y),
                        'class_id': class_id,
                        'class_name': class_name,
                        'waste_type': waste_type,
                        'confidence': confidence,
                        'timestamp': time.time()
                    }
                    
                    detections.append(detection)
                    
                    # Update statistics
                    self.stats['total_detections'] += 1
                    if waste_type == 'plastic_bottle':
                        self.stats['plastic_bottles'] += 1
                    elif waste_type == 'aluminum_can':
                        self.stats['aluminum_cans'] += 1
                    elif waste_type == 'juice_box':
                        self.stats['juice_boxes'] += 1
        
        return detections
    
    def _classify_waste(self, class_id, class_name):
        """Classify object as waste type"""
        # Check if it's a target class
        if class_id in self.target_classes:
            return self.target_classes[class_id]
        
        # Check by name (for both pretrained and custom models)
        class_lower = class_name.lower()
        
        if 'bottle' in class_lower or 'plastic' in class_lower:
            return 'plastic_bottle'
        elif 'can' in class_lower or 'aluminum' in class_lower or 'tin' in class_lower:
            return 'aluminum_can'
        elif 'box' in class_lower or 'juice' in class_lower or 'carton' in class_lower:
            return 'juice_box'
        
        return None  # Not a waste object we care about
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center = det['center']
            waste_type = det['waste_type']
            confidence = det['confidence']
            
            # Get color for this waste type
            color = self.colors.get(waste_type, self.colors['unknown'])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw center point with crosshair
            cv2.circle(frame, center, 6, color, -1)
            cv2.circle(frame, center, 10, color, 2)
            cv2.line(frame, (center[0]-15, center[1]), (center[0]+15, center[1]), color, 2)
            cv2.line(frame, (center[0], center[1]-15), (center[0], center[1]+15), color, 2)
            
            # Prepare label
            label = waste_type.replace('_', ' ').title()
            conf_text = f"{confidence:.2f}"
            coord_text = f"({center[0]}, {center[1]})"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-35), (x1 + label_size[0] + 80, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1+5, y1-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw coordinates below box
            cv2.putText(frame, coord_text, (center[0]-40, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def draw_info_overlay(self, frame, detections, fps):
        """Draw information overlay on frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "WASTE DETECTION SYSTEM", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Detection count
        cv2.putText(frame, f"Detections: {len(detections)}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Count by category
        bottles = sum(1 for d in detections if d['waste_type'] == 'plastic_bottle')
        cans = sum(1 for d in detections if d['waste_type'] == 'aluminum_can')
        boxes = sum(1 for d in detections if d['waste_type'] == 'juice_box')
        
        cv2.putText(frame, f"Bottles: {bottles}", (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['plastic_bottle'], 2)
        cv2.putText(frame, f"Cans: {cans}", (150, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['aluminum_can'], 2)
        cv2.putText(frame, f"Boxes: {boxes}", (260, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['juice_box'], 2)
        
        # Confidence threshold
        cv2.putText(frame, f"Threshold: {self.confidence_threshold:.2f}", (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Legend
        legend_y = h - 80
        cv2.rectangle(frame, (10, legend_y-10), (w-10, h-10), (0, 0, 0), -1)
        cv2.putText(frame, "Controls: Q-Quit | S-Screenshot | P-Pause | R-Report | +/- Threshold",
                   (20, legend_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_detection_log(self, filename=None):
        """Save detection log to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/detection_log_{timestamp}.json"
        
        os.makedirs('outputs', exist_ok=True)
        
        data = {
            'session_info': {
                'date': datetime.now().isoformat(),
                'model': str(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else 'yolov8n.pt',
                'confidence_threshold': self.confidence_threshold,
                'total_frames': self.frame_count
            },
            'statistics': self.stats,
            'detections': self.detection_log[-100:]  # Last 100 detections
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Detection log saved: {filename}")
        return filename
    
    def print_statistics(self):
        """Print detection statistics"""
        print("\n" + "="*70)
        print("DETECTION STATISTICS")
        print("="*70)
        
        if self.frame_count == 0:
            print("No frames processed yet")
            return
        
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        detection_rate = self.stats['total_detections'] / self.frame_count if self.frame_count > 0 else 0
        
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Total Detections: {self.stats['total_detections']}")
        print(f"Detection Rate: {detection_rate:.2f} objects/frame")
        print(f"\nDetections by Category:")
        print(f"  Plastic Bottles: {self.stats['plastic_bottles']}")
        print(f"  Aluminum Cans: {self.stats['aluminum_cans']}")
        print(f"  Juice Boxes: {self.stats['juice_boxes']}")
        
        if self.stats['total_detections'] > 0:
            print(f"\nBreakdown:")
            total = self.stats['total_detections']
            print(f"  Bottles: {(self.stats['plastic_bottles']/total)*100:.1f}%")
            print(f"  Cans: {(self.stats['aluminum_cans']/total)*100:.1f}%")
            print(f"  Boxes: {(self.stats['juice_boxes']/total)*100:.1f}%")
        
        print("="*70 + "\n")
    
    def run_camera(self, camera_id=0, record_video=False):
        """
        Run real-time detection from camera
        
        Args:
            camera_id: Camera device ID (usually 0)
            record_video: If True, saves video to outputs/
        """
        print("\n[STARTING CAMERA MODE]")
        print(f"Camera ID: {camera_id}")
        print(f"Recording: {'Yes' if record_video else 'No'}")
        print("\nPress 'q' to quit\n")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("✗ ERROR: Cannot open camera!")
            print("Try: ls /dev/video* to see available cameras")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera opened: {width}x{height}")
        
        # Video writer
        video_writer = None
        if record_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"outputs/detection_{timestamp}.mp4"
            os.makedirs('outputs', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
            print(f"✓ Recording to: {video_path}")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("✗ Failed to grab frame")
                        break
                    
                    # Measure processing time
                    start_time = time.time()
                    
                    # Detect objects
                    detections = self.detect_objects(frame)
                    
                    # Log detections
                    for det in detections:
                        self.detection_log.append({
                            'frame': self.frame_count,
                            'waste_type': det['waste_type'],
                            'confidence': det['confidence'],
                            'position': det['center']
                        })
                    
                    # Calculate FPS
                    processing_time = time.time() - start_time
                    fps = 1.0 / processing_time if processing_time > 0 else 0
                    self.fps_history.append(fps)
                    avg_fps = np.mean(self.fps_history)
                    
                    # Visualize
                    frame = self.draw_detections(frame, detections)
                    frame = self.draw_info_overlay(frame, detections, avg_fps)
                    
                    # Record frame
                    if video_writer:
                        video_writer.write(frame)
                    
                    self.frame_count += 1
                    self.stats['total_frames'] = self.frame_count
                
                # Display
                cv2.imshow('Waste Detection System - CV Only', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n[QUITTING]")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"outputs/screenshot_{timestamp}.jpg"
                    os.makedirs('outputs', exist_ok=True)
                    cv2.imwrite(filename, frame)
                    print(f"✓ Screenshot saved: {filename}")
                elif key == ord('p'):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMED"
                    print(f"[{status}]")
                elif key == ord('r'):
                    self.print_statistics()
                elif key == ord('+') or key == ord('='):
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            print("\n[SESSION COMPLETE]")
            self.print_statistics()
            
            # Save log
            self.save_detection_log()


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Waste Detection CV System')
    parser.add_argument('--model', default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--record', action='store_true', help='Record video')
    
    args = parser.parse_args()
    
    # Initialize system
    cv_system = WasteDetectionCV(
        model_path=args.model,
        confidence_threshold=args.conf
    )
    
    # Run detection
    cv_system.run_camera(
        camera_id=args.camera,
        record_video=args.record
    )
