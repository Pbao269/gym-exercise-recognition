"""
API Client example for Gym Exercise Recognition API
Demonstrates how to interact with the API securely.
"""

import requests
import json
import time
import numpy as np
from typing import Dict, List, Optional

class GymExerciseAPIClient:
    """Client for interacting with the Gym Exercise Recognition API."""
    
    def __init__(self, base_url: str, api_key: str):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API (e.g., "http://localhost:8000")
            api_key: Your API authentication key
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def health_check(self) -> Dict:
        """Check API health status."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def get_supported_exercises(self) -> List[str]:
        """Get list of supported exercises."""
        try:
            response = self.session.get(f"{self.base_url}/exercises")
            response.raise_for_status()
            return response.json()["exercises"]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching exercises: {e}")
            return []
    
    def predict_exercise(
        self, 
        sensor_data: Dict[str, List[float]], 
        include_probabilities: bool = False,
        include_features: bool = False
    ) -> Optional[Dict]:
        """
        Predict exercise from sensor data.
        
        Args:
            sensor_data: Dictionary with sensor readings (8 sensors × 60 time steps)
            include_probabilities: Include prediction probabilities for all classes
            include_features: Include extracted statistical features
            
        Returns:
            Prediction response or None if error
        """
        payload = {
            "sensor_data": sensor_data,
            "include_probabilities": include_probabilities,
            "include_features": include_features
        }
        
        try:
            response = self.session.post(f"{self.base_url}/predict", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Prediction error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
    
    def get_model_info(self) -> Optional[Dict]:
        """Get model information."""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching model info: {e}")
            return None

def generate_sample_data() -> Dict[str, List[float]]:
    """Generate sample sensor data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Simulate different exercise patterns
    time_steps = 60
    
    # Base patterns for different sensors
    session = [1.0] * time_steps
    
    # Simulate walking pattern (periodic acceleration)
    t = np.linspace(0, 3, time_steps)  # 3 seconds
    walking_freq = 2.0  # 2 Hz walking frequency
    
    a_x = 0.5 * np.sin(2 * np.pi * walking_freq * t) + np.random.normal(0, 0.1, time_steps)
    a_y = 0.3 * np.cos(2 * np.pi * walking_freq * t) + np.random.normal(0, 0.1, time_steps)
    a_z = 9.8 + 0.2 * np.sin(2 * np.pi * walking_freq * t) + np.random.normal(0, 0.1, time_steps)
    
    g_x = 0.1 * np.sin(2 * np.pi * walking_freq * t + np.pi/4) + np.random.normal(0, 0.05, time_steps)
    g_y = 0.1 * np.cos(2 * np.pi * walking_freq * t + np.pi/4) + np.random.normal(0, 0.05, time_steps)
    g_z = 0.05 * np.sin(4 * np.pi * walking_freq * t) + np.random.normal(0, 0.05, time_steps)
    
    c_1 = 0.5 + 0.1 * np.sin(2 * np.pi * walking_freq * t) + np.random.normal(0, 0.02, time_steps)
    
    return {
        "session": session,
        "a_x": a_x.tolist(),
        "a_y": a_y.tolist(),
        "a_z": a_z.tolist(),
        "g_x": g_x.tolist(),
        "g_y": g_y.tolist(),
        "g_z": g_z.tolist(),
        "c_1": c_1.tolist()
    }

def main():
    """Example usage of the API client."""
    # Configuration
    API_URL = "http://localhost:8000"
    API_KEY = "demo_key_123"  # Use your actual API key
    
    # Initialize client
    client = GymExerciseAPIClient(API_URL, API_KEY)
    
    print("🏋️ Gym Exercise Recognition API Client Demo")
    print("=" * 50)
    
    # Health check
    print("1. Health Check:")
    health = client.health_check()
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Model Loaded: {health.get('model_loaded', False)}")
    print()
    
    # Get supported exercises
    print("2. Supported Exercises:")
    exercises = client.get_supported_exercises()
    if exercises:
        for i, exercise in enumerate(exercises, 1):
            print(f"   {i:2d}. {exercise}")
    print()
    
    # Get model info
    print("3. Model Information:")
    model_info = client.get_model_info()
    if model_info:
        print(f"   Type: {model_info.get('model_type', 'N/A')}")
        print(f"   Accuracy: {model_info.get('accuracy', 'N/A')}")
        print(f"   F1-Score: {model_info.get('f1_score', 'N/A')}")
    print()
    
    # Make predictions
    print("4. Exercise Prediction:")
    sample_data = generate_sample_data()
    
    # Basic prediction
    print("   Basic prediction...")
    start_time = time.time()
    result = client.predict_exercise(sample_data)
    end_time = time.time()
    
    if result:
        print(f"   Exercise: {result['exercise']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   API Processing Time: {result['processing_time_ms']:.2f} ms")
        print(f"   Total Round-trip Time: {(end_time - start_time) * 1000:.2f} ms")
    print()
    
    # Detailed prediction with probabilities
    print("5. Detailed Prediction with Probabilities:")
    detailed_result = client.predict_exercise(
        sample_data, 
        include_probabilities=True
    )
    
    if detailed_result and detailed_result.get('probabilities'):
        print("   Top 5 Predictions:")
        probs = detailed_result['probabilities']
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        for i, (exercise, prob) in enumerate(sorted_probs[:5], 1):
            print(f"   {i}. {exercise:<15} {prob:.3f}")
    print()
    
    print("✅ API Client Demo Completed!")

if __name__ == "__main__":
    main()
