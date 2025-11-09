#!/usr/bin/env python3
"""
Simple Dataset Downloader for Waste Detection
No API keys required for some datasets
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import requests
        return True
    except ImportError:
        print("Installing requests...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        return True

def download_from_roboflow_public():
    """Download from public Roboflow datasets"""
    print("\n" + "="*70)
    print("ROBOFLOW PUBLIC DATASETS")
    print("="*70)
    
    print("\nOption 1: Manual Download (Recommended)")
    print("-" * 70)
    print("1. Visit: https://universe.roboflow.com")
    print("2. Search for: 'waste classification' or 'bottle detection'")
    print("3. Popular datasets:")
    print("   - 'Waste Classification' by Joseph Nelson")
    print("   - 'Bottle Detection'")
    print("   - 'Recyclable Waste Detection'")
    print("4. Click 'Download' → Select 'YOLOv8' format")
    print("5. Download ZIP file")
    print("6. Extract to: ~/waste_sorting_project/dataset/")
    
    print("\nOption 2: With API Key (Free Account Required)")
    print("-" * 70)
    api_key = input("Enter Roboflow API key (or press Enter to skip): ").strip()
    
    if api_key:
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            
            print("\nSearching for waste detection datasets...")
            # Example: Download a specific project
            project = rf.workspace("waste-detection").project("waste-classification")
            dataset = project.version(1).download("yolov8")
            
            print(f"\n✓ Dataset downloaded to: {dataset.location}")
            return dataset.location
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Please use manual download method")
    
    return None

def download_from_kaggle():
    """Download from Kaggle"""
    print("\n" + "="*70)
    print("KAGGLE DATASETS")
    print("="*70)
    
    print("\nPopular Waste Detection Datasets on Kaggle:")
    print("-" * 70)
    print("1. Garbage Classification (12 classes)")
    print("   URL: kaggle.com/datasets/asdasdasasdas/garbage-classification")
    print("\n2. Waste Classification Data")
    print("   URL: kaggle.com/datasets/techsash/waste-classification-data")
    print("\n3. Drinking Waste Classification")
    print("   URL: kaggle.com/datasets/arkadiyhacks/drinking-waste-classification")
    
    print("\n\nTo Download:")
    print("1. Create free Kaggle account")
    print("2. Install Kaggle CLI: pip install kaggle")
    print("3. Get API token from kaggle.com/settings")
    print("4. Download: kaggle datasets download -d DATASET_NAME")
    
    has_kaggle = input("\nDo you have Kaggle CLI setup? (y/n): ").strip().lower()
    
    if has_kaggle == 'y':
        dataset_name = input("Enter dataset name (e.g., 'asdasdasasdas/garbage-classification'): ").strip()
        
        if dataset_name:
            try:
                print(f"\nDownloading {dataset_name}...")
                os.makedirs('dataset', exist_ok=True)
                subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', 'dataset'])
                
                # Unzip
                import zipfile
                zip_files = [f for f in os.listdir('dataset') if f.endswith('.zip')]
                if zip_files:
                    print(f"Extracting {zip_files[0]}...")
                    with zipfile.ZipFile(f'dataset/{zip_files[0]}', 'r') as zip_ref:
                        zip_ref.extractall('dataset/')
                    print("✓ Dataset extracted")
                
            except Exception as e:
                print(f"✗ Error: {e}")

def create_sample_dataset_structure():
    """Create empty dataset structure"""
    print("\n" + "="*70)
    print("CREATING DATASET STRUCTURE")
    print("="*70)
    
    dirs = [
        'dataset/images/train',
        'dataset/images/val',
        'dataset/images/test',
        'dataset/labels/train',
        'dataset/labels/val',
        'dataset/labels/test'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Created: {d}")
    
    # Create data.yaml
    yaml_content = """# Waste Sorting Dataset Configuration
# Path: relative to this file location

path: ./dataset
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: plastic_bottle
  1: aluminum_can
  2: juice_box

# Number of classes
nc: 3

# Notes:
# - Add images to images/train, images/val, images/test
# - Add corresponding labels (YOLO format) to labels/train, labels/val, labels/test
# - Label format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
"""
    
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("\n✓ Created data.yaml")
    print("\nDataset structure ready!")
    print("Next: Add images and labels to the folders")

def show_labeling_tools():
    """Show image labeling tool options"""
    print("\n" + "="*70)
    print("IMAGE LABELING TOOLS")
    print("="*70)
    
    print("\n1. LabelImg (Desktop - Easy)")
    print("   Install: pip install labelImg")
    print("   Run: labelImg")
    print("   Features: Draw boxes, assign classes, save YOLO format")
    
    print("\n2. Roboflow (Web - Easiest)")
    print("   URL: https://roboflow.com")
    print("   Features: Cloud-based, auto-labeling, augmentation")
    
    print("\n3. CVAT (Web/Desktop - Advanced)")
    print("   URL: https://www.cvat.ai")
    print("   Features: Team collaboration, video annotation")
    
    print("\n4. Label Studio (Desktop - Open Source)")
    print("   Install: pip install label-studio")
    print("   Features: ML-assisted labeling, multiple formats")

def main():
    print("="*70)
    print("WASTE DETECTION DATASET SETUP")
    print("Student ID: 240108542")
    print("="*70)
    
    check_dependencies()
    
    print("\nChoose an option:")
    print("1. Download from Roboflow")
    print("2. Download from Kaggle")
    print("3. Create empty dataset structure (add your own data)")
    print("4. Show image labeling tools")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        download_from_roboflow_public()
    elif choice == '2':
        download_from_kaggle()
    elif choice == '3':
        create_sample_dataset_structure()
    elif choice == '4':
        show_labeling_tools()
    elif choice == '5':
        print("Exiting...")
        return
    else:
        print("Invalid choice!")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. If you downloaded dataset → Run training:")
    print("   python3 train_model.py")
    print("\n2. If you created structure → Add images and labels, then train")
    print("\n3. Test current system with pretrained model:")
    print("   python3 cv_only_system.py")
    print("="*70)

if __name__ == "__main__":
    main()
