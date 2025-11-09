#!/usr/bin/env python3
"""
Training Script for Waste Detection Model
Simple and straightforward - just run and wait!
"""

from ultralytics import YOLO
import torch
import os
import yaml

def check_dataset(data_yaml='data.yaml'):
    """Verify dataset before training"""
    print("\n" + "="*70)
    print("CHECKING DATASET")
    print("="*70)
    
    if not os.path.exists(data_yaml):
        print(f"✗ ERROR: {data_yaml} not found!")
        print("\nCreate it with:")
        print("  python3 simple_dataset_downloader.py")
        return False
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"✓ Dataset config loaded")
    print(f"  Classes: {data.get('nc', 0)}")
    print(f"  Names: {data.get('names', [])}")
    
    # Check paths
    base_path = data.get('path', '.')
    train_path = os.path.join(base_path, data.get('train', ''))
    
    if os.path.exists(train_path):
        num_images = len([f for f in os.listdir(train_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  Training images: {num_images}")
        
        if num_images == 0:
            print("✗ ERROR: No training images found!")
            print(f"  Add images to: {train_path}")
            return False
        elif num_images < 50:
            print("⚠️  WARNING: Very few images. Recommend 100+ per class")
        
        print("✓ Dataset looks good!")
        return True
    else:
        print(f"✗ ERROR: Training path not found: {train_path}")
        return False

def train_model(data_yaml='data.yaml', epochs=100, model_size='n'):
    """
    Train the waste detection model
    
    Args:
        data_yaml: Path to dataset config
        epochs: Number of training epochs (100-200 recommended)
        model_size: 'n' (fastest), 's' (balanced), 'm' (accurate)
    """
    
    print("\n" + "="*70)
    print("WASTE DETECTION MODEL TRAINING")
    print("Student ID: 240108542")
    print("="*70)
    
    # Check dataset first
    if not check_dataset(data_yaml):
        return None
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Device: {device}")
    
    if device == 'cpu':
        print("⚠️  Training on CPU will be SLOW (1-3 hours)")
        print("   Consider: Reducing epochs to 50, using nano model")
        proceed = input("\nContinue? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Training cancelled")
            return None
    
    # Load pretrained model
    model_name = f'yolov8{model_size}.pt'
    print(f"\n✓ Loading {model_name}...")
    model = YOLO(model_name)
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Model: YOLOv8-{model_size}")
    print(f"  Batch: Auto (optimized)")
    print(f"  Image size: 640x640")
    
    # Estimate time
    est_minutes = epochs * (2 if device == 'cpu' else 0.3)
    print(f"\n⏱️  Estimated time: {est_minutes:.0f}-{est_minutes*1.5:.0f} minutes")
    
    input("\nPress ENTER to start training...")
    
    print("\n" + "="*70)
    print("TRAINING STARTED - Please wait...")
    print("="*70 + "\n")
    
    try:
        # Train the model
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=-1,  # Auto batch size
            name='waste_detection',
            patience=20,  # Early stopping
            save=True,
            device=device,
            
            # Optimization
            optimizer='AdamW',
            lr0=0.001,
            
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            
            # Performance
            cache=True,
            workers=4,
            verbose=True,
            plots=True
        )
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)
        
        # Validate
        print("\nRunning validation...")
        metrics = model.val()
        
        # Display results
        print(f"\n📊 RESULTS:")
        print(f"  mAP50: {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
        print(f"  mAP50-95: {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
        print(f"  Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
        print(f"  Recall: {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")
        
        # Check if target met
        if metrics.box.map50 >= 0.85:
            print("\n🎉 SUCCESS! Achieved 85%+ mAP50 target!")
        else:
            print(f"\n⚠️  Target: 85%+, Current: {metrics.box.map50*100:.2f}%")
            print("   Tips to improve:")
            print("   - Collect more training images (aim for 200+ per class)")
            print("   - Train for more epochs (150-200)")
            print("   - Use larger model (yolov8s or yolov8m)")
            print("   - Check data quality (clear images, accurate labels)")
        
        # Save location
        best_model = 'runs/detect/waste_detection/weights/best.pt'
        print(f"\n📦 Model saved: {best_model}")
        print(f"📊 Results: runs/detect/waste_detection/")
        print(f"📈 Plots: runs/detect/waste_detection/*.png")
        
        return best_model
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
        return None
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        return None

def test_trained_model(model_path):
    """Quick test of trained model"""
    print("\n" + "="*70)
    print("TESTING TRAINED MODEL")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return
    
    print("\nStarting camera test...")
    print("Press 'q' to quit\n")
    
    import cv2
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot open camera")
        return
    
    frame_count = 0
    detection_count = 0
    
    while frame_count < 200:  # Test for 200 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        results = model(frame, conf=0.5)
        detection_count += len(results[0].boxes)
        
        # Visualize
        annotated = results[0].plot()
        
        # Info
        cv2.putText(annotated, f"Frame: {frame_count} | Detections: {detection_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Model Test - Press Q to quit', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    avg_detections = detection_count / frame_count if frame_count > 0 else 0
    print(f"\n✓ Test complete!")
    print(f"  Frames: {frame_count}")
    print(f"  Total detections: {detection_count}")
    print(f"  Avg per frame: {avg_detections:.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Waste Detection Model')
    parser.add_argument('--data', default='data.yaml', help='Dataset config file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--model', default='n', choices=['n', 's', 'm', 'l'],
                       help='Model size: n=nano(fast), s=small, m=medium, l=large')
    parser.add_argument('--test', action='store_true', help='Test model after training')
    
    args = parser.parse_args()
    
    print("\n🚀 Starting training process...\n")
    
    # Train model
    model_path = train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        model_size=args.model
    )
    
    # Test if requested
    if model_path and args.test:
        test_trained_model(model_path)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. Test your trained model:")
    print("   python3 cv_only_system.py --model runs/detect/waste_detection/weights/best.pt")
    print("\n2. View training plots:")
    print("   Open: runs/detect/waste_detection/results.png")
    print("\n3. Check confusion matrix:")
    print("   Open: runs/detect/waste_detection/confusion_matrix.png")
    print("\n" + "="*70)
