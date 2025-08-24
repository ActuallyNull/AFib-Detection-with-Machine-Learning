#!/usr/bin/env python3
"""
Test script to verify the React + FastAPI ECG classification setup.
"""

import requests
import json
import time
import os
import sys
from pathlib import Path

def test_backend_health():
    """Test backend health endpoint"""
    print("🔍 Testing backend health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend health check passed: {data}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend health check failed: {e}")
        return False

def test_model_info():
    """Test model information endpoint"""
    print("🔍 Testing model info...")
    try:
        response = requests.get("http://localhost:8000/model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved: {data}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info failed: {e}")
        return False

def test_frontend():
    """Test frontend accessibility"""
    print("🔍 Testing frontend...")
    try:
        response = requests.get("http://localhost:3000", timeout=10)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend check failed: {e}")
        return False

def test_api_docs():
    """Test API documentation"""
    print("🔍 Testing API documentation...")
    try:
        response = requests.get("http://localhost:8000/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation is accessible")
            return True
        else:
            print(f"❌ API docs check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API docs check failed: {e}")
        return False

def create_test_ecg_data():
    """Create a simple test ECG signal"""
    import numpy as np
    
    # Create a simple sine wave as test ECG data
    fs = 300  # 300 Hz sampling rate
    duration = 10  # 10 seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Create a realistic ECG-like signal
    signal = (
        0.5 * np.sin(2 * np.pi * 1.2 * t) +  # Heart rate ~72 bpm
        0.3 * np.sin(2 * np.pi * 2.4 * t) +  # Second harmonic
        0.1 * np.random.randn(len(t))        # Noise
    )
    
    return signal, fs

def test_prediction_endpoint():
    """Test prediction endpoint with synthetic data"""
    print("🔍 Testing prediction endpoint...")
    
    try:
        # Create test data
        signal, fs = create_test_ecg_data()
        
        # Save as CSV for testing
        test_file = "test_ecg.csv"
        import pandas as pd
        df = pd.DataFrame({"ecg": signal})
        df.to_csv(test_file, index=False)
        
        # Test prediction
        with open(test_file, "rb") as f:
            files = {"file": ("test_ecg.csv", f, "text/csv")}
            response = requests.post("http://localhost:8000/predict", files=files, timeout=30)
        
        # Clean up test file
        os.remove(test_file)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful: {data}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def test_preprocess_endpoint():
    """Test preprocessing endpoint"""
    print("🔍 Testing preprocessing endpoint...")
    
    try:
        # Create test data
        signal, fs = create_test_ecg_data()
        
        # Save as CSV for testing
        test_file = "test_ecg_preprocess.csv"
        import pandas as pd
        df = pd.DataFrame({"ecg": signal})
        df.to_csv(test_file, index=False)
        
        # Test preprocessing
        with open(test_file, "rb") as f:
            files = {"file": ("test_ecg_preprocess.csv", f, "text/csv")}
            response = requests.post("http://localhost:8000/preprocess", files=files, timeout=30)
        
        # Clean up test file
        os.remove(test_file)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Preprocessing successful: {len(data['signal'])} samples")
            return True
        else:
            print(f"❌ Preprocessing failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def check_docker_services():
    """Check if Docker services are running"""
    print("🔍 Checking Docker services...")
    
    try:
        import subprocess
        result = subprocess.run(["docker-compose", "ps"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Docker Compose services:")
            print(result.stdout)
            return True
        else:
            print("❌ Docker Compose check failed")
            return False
    except FileNotFoundError:
        print("⚠️  Docker Compose not found - skipping service check")
        return True
    except Exception as e:
        print(f"❌ Docker service check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 ECG Classification AI - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Docker Services", check_docker_services),
        ("Backend Health", test_backend_health),
        ("Model Info", test_model_info),
        ("API Documentation", test_api_docs),
        ("Frontend", test_frontend),
        ("Preprocessing", test_preprocess_endpoint),
        ("Prediction", test_prediction_endpoint),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is working correctly.")
        print("\n🌐 Access your application:")
        print("   Frontend: http://localhost:3000")
        print("   Backend API: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Ensure Docker Compose is running: docker-compose up -d")
        print("   2. Check if ports 3000 and 8000 are available")
        print("   3. Verify model.keras file exists in the project root")
        print("   4. Check Docker logs: docker-compose logs")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

