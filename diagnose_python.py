#!/usr/bin/env python3
"""
Diagnose Python environment issues
Finds where packages are installed and why they might not be detected
"""

import sys
import subprocess
import os

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def check_python_versions():
    """Check all Python versions on system"""
    print_section("PYTHON VERSIONS")
    
    commands = ['python --version', 'python3 --version', 'python3.11 --version']
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {cmd}: {result.stdout.strip()}")
            else:
                print(f"✗ {cmd}: Not found")
        except:
            print(f"✗ {cmd}: Not available")
    
    print(f"\nCurrent Python executable: {sys.executable}")
    print(f"Current Python version: {sys.version}")

def check_pip_versions():
    """Check pip installations"""
    print_section("PIP VERSIONS")
    
    commands = ['pip --version', 'pip3 --version']
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {cmd}:")
                print(f"  {result.stdout.strip()}")
            else:
                print(f"✗ {cmd}: Not found")
        except:
            print(f"✗ {cmd}: Not available")

def check_package_locations():
    """Check where packages are installed"""
    print_section("PACKAGE INSTALLATION LOCATIONS")
    
    packages = ['opencv-python', 'ultralytics', 'numpy', 'torch']
    
    for package in packages:
        print(f"\n{package}:")
        try:
            # Using pip show
            result = subprocess.run(['pip3', 'show', package], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Location:' in line or 'Version:' in line:
                        print(f"  {line}")
            else:
                print(f"  ✗ Not found with pip3")
        except:
            print(f"  ✗ Error checking package")

def check_python_path():
    """Check Python path"""
    print_section("PYTHON PATH")
    
    print("sys.path entries:")
    for i, path in enumerate(sys.path, 1):
        print(f"  {i}. {path}")

def test_imports():
    """Test importing packages"""
    print_section("IMPORT TESTS")
    
    packages = {
        'cv2': 'OpenCV',
        'ultralytics': 'Ultralytics YOLO',
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'PIL': 'Pillow',
        'torchvision': 'TorchVision'
    }
    
    failed = []
    
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name} ({module})")
        except ImportError as e:
            print(f"✗ {name} ({module})")
            print(f"  Error: {e}")
            failed.append((module, name))
    
    return failed

def suggest_fixes(failed_imports):
    """Suggest fixes based on failed imports"""
    print_section("SUGGESTED FIXES")
    
    if not failed_imports:
        print("✓ All packages import successfully!")
        return
    
    print("Some packages failed to import. Try these solutions:\n")
    
    # Get the Python executable path
    python_exe = sys.executable
    
    print("SOLUTION 1: Install with the EXACT Python you're using")
    print(f"Run this command:")
    print(f"  {python_exe} -m pip install --upgrade --force-reinstall ", end="")
    print(" ".join([pkg[0].replace('cv2', 'opencv-python').replace('PIL', 'Pillow') 
                    for pkg in failed_imports]))
    
    print("\n\nSOLUTION 2: Use virtual environment (recommended)")
    print("  python3 -m venv cv_env")
    print("  source cv_env/bin/activate")
    print("  pip install opencv-python ultralytics numpy torch torchvision Pillow")
    
    print("\n\nSOLUTION 3: Check if using conda")
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ Conda detected!")
            print("  You might be in a conda environment.")
            print("  Try: conda install -c conda-forge opencv")
            print("       pip install ultralytics")
    except:
        print("  (conda not detected)")
    
    print("\n\nSOLUTION 4: Ubuntu/Debian specific")
    print("  sudo apt update")
    print("  sudo apt install python3-opencv python3-pip")
    print("  python3 -m pip install --user ultralytics torch torchvision")

def check_opencv_specific():
    """Check OpenCV specific issues"""
    print_section("OPENCV SPECIFIC CHECKS")
    
    # Check if opencv-python is installed
    try:
        result = subprocess.run(['pip3', 'list'], capture_output=True, text=True)
        opencv_packages = [line for line in result.stdout.split('\n') if 'opencv' in line.lower()]
        
        if opencv_packages:
            print("OpenCV packages found:")
            for pkg in opencv_packages:
                print(f"  {pkg}")
            
            # Check for conflicts
            if len(opencv_packages) > 1:
                print("\n⚠️  WARNING: Multiple OpenCV packages detected!")
                print("   This can cause conflicts.")
                print("\n   Fix: Uninstall all and reinstall one:")
                print("   pip3 uninstall opencv-python opencv-contrib-python opencv-python-headless")
                print("   pip3 install opencv-python")
        else:
            print("✗ No OpenCV packages found")
    except:
        print("✗ Error checking OpenCV packages")
    
    # Try importing cv2
    try:
        import cv2
        print(f"\n✓ OpenCV imports successfully")
        print(f"  Version: {cv2.__version__}")
    except ImportError as e:
        print(f"\n✗ OpenCV import failed: {e}")

def main():
    print("=" * 70)
    print("PYTHON ENVIRONMENT DIAGNOSTIC TOOL")
    print("=" * 70)
    
    check_python_versions()
    check_pip_versions()
    check_package_locations()
    check_python_path()
    check_opencv_specific()
    
    failed = test_imports()
    suggest_fixes(failed)
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    
    if not failed:
        print("\n🎉 All packages are working!")
        print("\nYou can now run:")
        print("  python3 cv_only_system.py")
    else:
        print(f"\n⚠️  {len(failed)} package(s) need attention")
        print("\nFollow the suggested fixes above, then run this diagnostic again.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
