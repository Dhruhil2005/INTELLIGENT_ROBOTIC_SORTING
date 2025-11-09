#!/usr/bin/env python3
"""
Verify that everything is set up correctly
Run this after creating the project structure
"""

import sys
import os

def check_directories():
    """Check if required directories exist"""
    print("\n[1/4] Checking Directories...")
    
    required_dirs = [
        'dataset', 'models', 'outputs', 'tests',
        'dataset/images', 'dataset/labels'
    ]
    
    missing = []
    for d in required_dirs:
        if os.path.exists(d):
            print(f"  ✓ {d}")
        else:
            print(f"  ✗ {d} (missing)")
            missing.append(d)
    
    if missing:
        print("\n  Creating missing directories...")
        for d in missing:
            os.makedirs(d, exist_ok=True)
            print(f"  ✓ Created: {d}")
    
    return len(missing) == 0

def check_python_packages():
    """Check if required Python packages are installed"""
    print("\n[2/4] Checking Python Packages...")
    
    packages = {
        'cv2': 'opencv-python',
        'ultralytics': 'ultralytics',
        'numpy': 'numpy',
        'torch': 'torch',
        'PIL': 'Pillow'
    }
    
    missing = []
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {pip_name}")
        except ImportError:
            print(f"  ✗ {pip_name} (not installed)")
            missing.append(pip_name)
    
    if missing:
        print("\n  To install missing packages, run:")
        print(f"  pip3 install {' '.join(missing)}")
        return False
    
    return True

def check_camera():
    """Check if camera is accessible"""
    print("\n[3/4] Checking Camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  ✓ Camera working: {w}x{h}")
                cap.release()
                return True
            else:
                print("  ✗ Camera opened but cannot read frames")
                cap.release()
                return False
        else:
            print("  ✗ Cannot open camera")
            print("    Try: ls /dev/video* to see available cameras")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def check_scripts():
    """Check if main scripts exist"""
    print("\n[4/4] Checking Python Scripts...")
    
    scripts = [
        'cv_only_system.py',
        'simple_dataset_downloader.py',
        'simple_training_script.py'
    ]
    
    missing = []
    for script in scripts:
        if os.path.exists(script):
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} (not found)")
            missing.append(script)
    
    if missing:
        print("\n  ⚠️  Missing scripts! You need to create them.")
        print("     See the artifacts in the chat above.")
    
    return len(missing) == 0

def download_yolo_model():
    """Download YOLOv8 pretrained model"""
    print("\n[BONUS] Downloading YOLOv8 Model...")
    
    try:
        from ultralytics import YOLO
        
        if os.path.exists('yolov8n.pt'):
            print("  ✓ yolov8n.pt already exists")
            return True
        
        print("  Downloading YOLOv8-nano model (6MB)...")
        model = YOLO('yolov8n.pt')
        print("  ✓ Model downloaded successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Error downloading model: {e}")
        return False

def main():
    print("=" * 70)
    print("WASTE DETECTION CV SYSTEM - SETUP VERIFICATION")
    print("Student ID: 240108542")
    print("=" * 70)
    
    results = {
        'directories': check_directories(),
        'packages': check_python_packages(),
        'camera': check_camera(),
        'scripts': check_scripts()
    }
    
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_good = True
    for name, status in results.items():
        status_text = "✓ PASS" if status else "✗ FAIL"
        print(f"  {name.capitalize()}: {status_text}")
        if not status:
            all_good = False
    
    print("=" * 70)
    
    if all_good:
        print("\n🎉 Everything looks good!")
        
        # Try to download model
        if results['packages']:
            download_yolo_model()
        
        print("\n📋 NEXT STEPS:")
        print("  1. Run: python3 cv_only_system.py")
        print("  2. Show bottles/cans to camera")
        print("  3. Press 'q' to quit, 's' for screenshot")
        
    else:
        print("\n⚠️  Some issues found. Please fix them first.")
        
        if not results['packages']:
            print("\n  Install packages:")
            print("  pip3 install opencv-python ultralytics numpy torch Pillow")
        
        if not results['scripts']:
            print("\n  Create missing Python scripts from the chat artifacts above")
        
        if not results['camera']:
            print("\n  Camera issues: Check connections or try different camera ID")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
