#!/usr/bin/env python3
"""
Training Script for Waste Detection - Optimized for 80%+ Accuracy
Includes data augmentation, hyperparameter tuning, and validation

Author: Dhruhil Gajera
Student ID: 240108542
"""

from ultralytics import YOLO
import torch
import os
import yaml
import matplotlib.pyplot as plt

def check_dataset(data_yaml='data.yaml'):
    """Verify dataset quality before training"""
    print("\n" + "="*70)
    print("DATASET VERIFICATION")
    print("="*70)
    
    if not os.path.exists(data_yaml):
        print(f"✗ ERROR: {data_yaml} not found!")
        return False
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"✓ Dataset config loaded")
    print(f"  Classes: {data.get('nc', 0)}")
    print(f"  Names: {data.get('names', [])}")
    
    # Check paths
    base_path = data.get('path', '.')
    train_path = os.path.join(base_path, data.get('train', 'images/train'))
    val_path = os.path.join(base_path, data.get('val', 'images/val'))
    
    # Count images
    if os.path.exists(train_path):
        train_images = len([f for f in os.listdir(train_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  Training images: {train_images}")
        
        if train_images < 100:
            print("  ⚠️  WARNING: Less than 100 images! Recommend 200+ per class")
            print("     For 3 classes, aim for 600+ total images")
        elif train_images < 300:
            print("  ⚠️  Moderate dataset size. 600+ recommended for best results")
        else:
            print("  ✓ Good dataset size!")
    else:
        print(f"  ✗ Training path not found: {train_path}")
        return False
    
    if os.path.exists(val_path):
        val_images = len([f for f in os.listdir(val_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  Validation images: {val_images}")
        
        if val_images < 30:
            print("  ⚠️  WARNING: Very few validation images")
    else:
        print(f"  ⚠️  Validation path not found: {val_path}")
    
    return train_images >= 50

def train_for_accuracy(data_yaml='data.yaml', 
                       target_map=0.80,
                       max_epochs=200,
                       model_size='s'):
    """
    Train model to achieve target accuracy
    
    Args:
        data_yaml: Dataset configuration file
        target_map: Target mAP50 (0.80 = 80%)
        max_epochs: Maximum training epochs
        model_size: 'n'=nano(fast), 's'=small(balanced), 'm'=medium(accurate)
    """
    
    print("\n" + "="*70)
    print("WASTE DETECTION TRAINING - TARGET 80%+ ACCURACY")
    print("Student ID: 240108542")
    print("="*70)
    
    # Verify dataset
    if not check_dataset(data_yaml):
        print("\n✗ Dataset verification failed!")
        return None
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Device: {device}")
    
    if device == 'cpu':
        print("⚠️  Training on CPU - will be SLOW!")
        print("   Recommend: Use Google Colab GPU (free)")
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != 'y':
            return None
    
    # Load model
    model_name = f'yolov8{model_size}.pt'
    print(f"\n✓ Loading {model_name}...")
    model = YOLO(model_name)
    
    # Training configuration for high accuracy
    print(f"\n📋 Training Configuration:")
    print(f"  Target mAP50: {target_map*100:.0f}%")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Model: YOLOv8-{model_size}")
    print(f"  Device: {device}")
    
    # Estimate time
    if device == 'cuda':
        est_time = max_epochs * 0.5  # ~30 sec per epoch on GPU
    else:
        est_time = max_epochs * 3    # ~3 min per epoch on CPU
    
    print(f"\n⏱️  Estimated time: {est_time/60:.1f} - {est_time/60*1.5:.1f} hours")
    
    input("\nPress ENTER to start training...")
    
    print("\n" + "="*70)
    print("🚀 TRAINING STARTED")
    print("="*70 + "\n")
    
    try:
        # Train with optimized hyperparameters
        results = model.train(
            data=data_yaml,
            epochs=max_epochs,
            imgsz=640,
            batch=-1,  # Auto-batch
            name='waste_detection_80pct',
            patience=50,  # Early stopping after 50 epochs no improvement
            save=True,
            device=device,
            
            # Optimizer settings for better convergence
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            
            # Data augmentation for robustness
            hsv_h=0.015,      # Hue augmentation
            hsv_s=0.7,        # Saturation
            hsv_v=0.4,        # Value/brightness
            degrees=15.0,     # Rotation
            translate=0.1,    # Translation
            scale=0.5,        # Scale
            shear=2.0,        # Shear
            perspective=0.0,  # Perspective
            flipud=0.0,       # Flip up-down (usually not for waste)
            fliplr=0.5,       # Flip left-right
            mosaic=1.0,       # Mosaic augmentation
            mixup=0.1,        # Mixup augmentation
            copy_paste=0.1,   # Copy-paste augmentation
            
            # Performance
            cache=True,       # Cache images in RAM
            workers=8,        # Data loading workers
            project='runs/detect',
            exist_ok=True,
            
            # Validation
            val=True,
            plots=True,
            verbose=True
        )
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)
        
        # Validate final model
        print("\n📊 Running final validation...")
        metrics = model.val()
        
        # Display results
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        precision = metrics.box.mp
        recall = metrics.box.mr
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"  mAP50:    {map50:.4f} ({map50*100:.2f}%)")
        print(f"  mAP50-95: {map50_95:.4f} ({map50_95*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        
        # Check if target achieved
        if map50 >= target_map:
            print(f"\n🎉 SUCCESS! Achieved {target_map*100:.0f}%+ mAP50 target!")
            print(f"   Your model: {map50*100:.2f}%")
        else:
            print(f"\n⚠️  Target: {target_map*100:.0f}%+, Current: {map50*100:.2f}%")
            print(f"   Gap: {(target_map - map50)*100:.2f}%")
            print("\n💡 Tips to improve:")
            print("   1. Collect more training images (aim for 600-1000+ total)")
            print("   2. Ensure balanced classes (equal images per class)")
            print("   3. Check label quality (accurate bounding boxes)")
            print("   4. Train longer (increase epochs to 300)")
            print("   5. Use larger model (yolov8m or yolov8l)")
            print("   6. Add more data augmentation")
        
        # Per-class performance
        if hasattr(metrics.box, 'ap_class_index'):
            print(f"\n📊 Per-Class Performance:")
            class_names = model.names
            for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
                print(f"   {class_names[idx]}: {ap*100:.2f}%")
        
        # Save paths
        best_model = 'runs/detect/waste_detection_80pct/weights/best.pt'
        print(f"\n📦 Best model: {best_model}")
        print(f"📊 Results: runs/detect/waste_detection_80pct/")
        
        # Generate report
        generate_training_report(metrics, best_model)
        
        return best_model
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
        return None
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_training_report(metrics, model_path):
    """Generate a training report"""
    print("\n" + "="*70)
    print("GENERATING TRAINING REPORT")
    print("="*70)
    
    report_file = 'outputs/training_report.txt'
    os.makedirs('outputs', exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("WASTE DETECTION MODEL - TRAINING REPORT\n")
        f.write("Student ID: 240108542\n")
        f.write("="*70 + "\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  mAP50:     {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)\n")
        f.write(f"  mAP50-95:  {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)\n")
        f.write(f"  Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)\n")
        f.write(f"  Recall:    {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)\n\n")
        
        f.write(f"MODEL PATH: {model_path}\n\n")
        
        f.write("NEXT STEPS:\n")
        f.write("  1. Test model: python3 cv_only_system.py --model " + model_path + "\n")
        f.write("  2. Check results: runs/detect/waste_detection_80pct/results.png\n")
        f.write("  3. View confusion matrix: runs/detect/waste_detection_80pct/confusion_matrix.png\n")
        f.write("  4. If accuracy < 80%, retrain with more data\n")
    
    print(f"✓ Report saved: {report_file}")

def quick_test_model(model_path):
    """Quick camera test of trained model"""
    print("\n" + "="*70)
    print("QUICK MODEL TEST")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return
    
    print(f"\nTesting model: {model_path}")
    print("Starting camera... Press 'q' to quit\n")
    
    import cv2
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot open camera")
        return
    
    frame_count = 0
    detection_count = 0
    
    while frame_count < 300:  # Test for 300 frames (~10 seconds)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        results = model(frame, conf=0.5)
        detections = len(results[0].boxes)
        detection_count += detections
        
        # Visualize
        annotated = results[0].plot()
        
        # Info
        cv2.putText(annotated, f"Frame: {frame_count} | Objects: {detections}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated, "Press 'q' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Model Test', annotated)
        
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
    
    parser = argparse.ArgumentParser(description='Train for 80%+ Accuracy')
    parser.add_argument('--data', default='data.yaml', help='Dataset config')
    parser.add_argument('--target', type=float, default=0.80, help='Target mAP50 (0.80=80%)')
    parser.add_argument('--epochs', type=int, default=200, help='Max epochs')
    parser.add_argument('--model', default='s', choices=['n', 's', 'm', 'l'],
                       help='Model size: n=nano, s=small, m=medium, l=large')
    parser.add_argument('--test', action='store_true', help='Test after training')
    
    args = parser.parse_args()
    
    print("\n🎯 Training for 80%+ Accuracy")
    print(f"Target: {args.target*100:.0f}%")
    print(f"Max epochs: {args.epochs}")
    print(f"Model: YOLOv8-{args.model}\n")
    
    # Train
    model_path = train_for_accuracy(
        data_yaml=args.data,
        target_map=args.target,
        max_epochs=args.epochs,
        model_size=args.model
    )
    
    # Test if requested
    if model_path and args.test:
        quick_test_model(model_path)
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
