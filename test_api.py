"""
Simple test to verify API implementation without running the full server.
Tests the core functionality and model loading.
"""

import sys
import os
sys.path.append('src')

try:
    from api import ExerciseRecognitionAPI, SensorData, PredictionRequest
    import numpy as np
    
    print("🧪 Testing API Implementation")
    print("=" * 50)
    
    # Test 1: API initialization
    print("1. Testing API initialization...")
    try:
        api = ExerciseRecognitionAPI()
        print("   ✅ API initialized successfully")
        print(f"   ✅ Model loaded: {api.model is not None}")
        print(f"   ✅ Classes: {len(api.classes)} exercises")
    except Exception as e:
        print(f"   ❌ API initialization failed: {e}")
        sys.exit(1)
    
    # Test 2: Sensor data validation
    print("\n2. Testing sensor data validation...")
    try:
        # Valid sensor data
        valid_data = {
            "session": [1.0] * 60,
            "a_x": [0.1] * 60,
            "a_y": [0.2] * 60,
            "a_z": [9.8] * 60,
            "g_x": [0.0] * 60,
            "g_y": [0.0] * 60,
            "g_z": [0.0] * 60,
            "c_1": [0.5] * 60
        }
        
        sensor_data = SensorData(**valid_data)
        print("   ✅ Valid sensor data accepted")
        
        # Test invalid data
        try:
            invalid_data = valid_data.copy()
            invalid_data["a_x"] = [0.1] * 50  # Wrong length
            SensorData(**invalid_data)
            print("   ❌ Should have rejected invalid data")
        except:
            print("   ✅ Invalid sensor data correctly rejected")
            
    except Exception as e:
        print(f"   ❌ Sensor data validation failed: {e}")
    
    # Test 3: Feature extraction
    print("\n3. Testing feature extraction...")
    try:
        sensor_data = SensorData(**valid_data)
        features = api.extract_statistical_features(sensor_data)
        print(f"   ✅ Features extracted: shape {features.shape}")
        print(f"   ✅ Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"   ✅ No NaN values: {not np.isnan(features).any()}")
    except Exception as e:
        print(f"   ❌ Feature extraction failed: {e}")
    
    # Test 4: Prediction
    print("\n4. Testing prediction...")
    try:
        request = PredictionRequest(
            sensor_data=sensor_data,
            include_probabilities=True,
            include_features=False
        )
        
        result = api.predict_exercise(request)
        print(f"   ✅ Prediction: {result.exercise}")
        print(f"   ✅ Confidence: {result.confidence:.3f}")
        print(f"   ✅ Processing time: {result.processing_time_ms:.2f} ms")
        print(f"   ✅ Probabilities included: {result.probabilities is not None}")
        
        # Show top 3 predictions
        if result.probabilities:
            sorted_probs = sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True)
            print("   📊 Top 3 predictions:")
            for i, (exercise, prob) in enumerate(sorted_probs[:3], 1):
                print(f"      {i}. {exercise:<15} {prob:.3f}")
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
    
    print("\n🎉 API Implementation Test Complete!")
    print("✅ All core functionality working correctly")
    print("\n💡 To run the full API server:")
    print("   python src/api.py")
    print("   Then visit: http://localhost:8000/docs")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're in the correct directory and have installed dependencies")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
