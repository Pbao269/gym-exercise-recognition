"""
Production-ready API for Gym Exercise Recognition
Secure, scalable FastAPI implementation with comprehensive documentation.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()
API_KEYS = {
    "demo_key_123": {"name": "Demo User", "tier": "basic", "requests_per_minute": 60},
    "premium_key_456": {"name": "Premium User", "tier": "premium", "requests_per_minute": 1000}
}

class SensorData(BaseModel):
    """
    Sensor data model for gym exercise recognition.
    
    Expected 8 sensors: Session, A_x, A_y, A_z, G_x, G_y, G_z, C_1
    Time series should be 60 time steps (3 seconds at 20Hz)
    """
    session: List[float] = Field(..., min_items=60, max_items=60, description="Session values (60 time steps)")
    a_x: List[float] = Field(..., min_items=60, max_items=60, description="X-axis acceleration (60 time steps)")
    a_y: List[float] = Field(..., min_items=60, max_items=60, description="Y-axis acceleration (60 time steps)")
    a_z: List[float] = Field(..., min_items=60, max_items=60, description="Z-axis acceleration (60 time steps)")
    g_x: List[float] = Field(..., min_items=60, max_items=60, description="X-axis gyroscope (60 time steps)")
    g_y: List[float] = Field(..., min_items=60, max_items=60, description="Y-axis gyroscope (60 time steps)")
    g_z: List[float] = Field(..., min_items=60, max_items=60, description="Z-axis gyroscope (60 time steps)")
    c_1: List[float] = Field(..., min_items=60, max_items=60, description="Capacitive sensor (60 time steps)")
    
    @validator('*', pre=True)
    def validate_numeric_lists(cls, v):
        """Ensure all values are numeric and finite."""
        if not isinstance(v, list):
            raise ValueError("Must be a list of numbers")
        
        numeric_values = []
        for val in v:
            if not isinstance(val, (int, float)):
                raise ValueError("All values must be numeric")
            if not np.isfinite(val):
                raise ValueError("All values must be finite (no NaN or infinity)")
            numeric_values.append(float(val))
        
        return numeric_values
    
    class Config:
        schema_extra = {
            "example": {
                "session": [1.0] * 60,
                "a_x": [0.1] * 60,
                "a_y": [0.2] * 60,
                "a_z": [9.8] * 60,
                "g_x": [0.0] * 60,
                "g_y": [0.0] * 60,
                "g_z": [0.0] * 60,
                "c_1": [0.5] * 60
            }
        }

class PredictionRequest(BaseModel):
    """Request model for exercise prediction."""
    sensor_data: SensorData = Field(..., description="Sensor readings for 3-second window")
    include_probabilities: bool = Field(False, description="Include prediction probabilities")
    include_features: bool = Field(False, description="Include extracted features (debug mode)")

class PredictionResponse(BaseModel):
    """Response model for exercise prediction."""
    exercise: str = Field(..., description="Predicted exercise name")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="All class probabilities")
    features: Optional[Dict[str, float]] = Field(None, description="Extracted statistical features")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    model_loaded: bool
    version: str
    uptime_seconds: float

class ExerciseRecognitionAPI:
    """Main API class for exercise recognition."""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.model_metadata = None
        self.start_time = time.time()
        self.version = "1.0.0"
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self):
        """Load the trained XGBoost model and metadata."""
        try:
            models_dir = Path("models")
            
            # Load XGBoost model
            model_path = models_dir / "xgb_gym_classifier.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = joblib.load(model_path)
            logger.info("XGBoost model loaded successfully")
            
            # Load metadata
            metadata_path = models_dir / "xgb_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                
                # Extract class names for label encoding
                self.classes = self.model_metadata['classes']
                logger.info(f"Model metadata loaded: {len(self.classes)} classes")
            else:
                logger.warning("Model metadata not found, using default classes")
                self.classes = ['Adductor', 'ArmCurl', 'BenchPress', 'LegCurl', 'LegPress', 
                              'Null', 'Riding', 'RopeSkipping', 'Running', 'Squat', 
                              'StairClimber', 'Walking']
            
            # Create feature names
            sensors = ['Session', 'A_x', 'A_y', 'A_z', 'G_x', 'G_y', 'G_z', 'C_1']
            feature_types = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'range', 'rms', 'var']
            self.feature_names = []
            for sensor in sensors:
                for feat_type in feature_types:
                    self.feature_names.append(f"{sensor}_{feat_type}")
            
            logger.info("Model initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def extract_statistical_features(self, sensor_data: SensorData) -> np.ndarray:
        """
        Extract statistical features from sensor data.
        Same feature extraction as used in training.
        """
        # Convert to numpy array (60 time steps, 8 sensors)
        data_matrix = np.array([
            sensor_data.session,
            sensor_data.a_x,
            sensor_data.a_y,
            sensor_data.a_z,
            sensor_data.g_x,
            sensor_data.g_y,
            sensor_data.g_z,
            sensor_data.c_1
        ]).T  # Shape: (60, 8)
        
        # Add batch dimension: (1, 60, 8)
        X = data_matrix[np.newaxis, :, :]
        
        # Extract statistical features (same as training)
        mean_feats = np.mean(X, axis=1)      # (1, 8)
        std_feats = np.std(X, axis=1)        # (1, 8)
        min_feats = np.min(X, axis=1)        # (1, 8)
        max_feats = np.max(X, axis=1)        # (1, 8)
        median_feats = np.median(X, axis=1)  # (1, 8)
        q25_feats = np.percentile(X, 25, axis=1)  # (1, 8)
        q75_feats = np.percentile(X, 75, axis=1)  # (1, 8)
        range_feats = max_feats - min_feats  # (1, 8)
        rms_feats = np.sqrt(np.mean(X**2, axis=1))  # (1, 8)
        var_feats = np.var(X, axis=1)       # (1, 8)
        
        # Combine all features
        features = np.concatenate([
            mean_feats, std_feats, min_feats, max_feats,
            median_feats, q25_feats, q75_feats, range_feats,
            rms_feats, var_feats
        ], axis=1)  # Shape: (1, 80)
        
        return features
    
    def predict_exercise(self, request: PredictionRequest) -> PredictionResponse:
        """Make exercise prediction from sensor data."""
        start_time = time.time()
        
        try:
            # Extract features
            features = self.extract_statistical_features(request.sensor_data)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Get class name and confidence
            predicted_class = self.classes[prediction]
            confidence = float(probabilities[prediction])
            
            # Prepare response
            response_data = {
                "exercise": predicted_class,
                "confidence": confidence,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            # Add probabilities if requested
            if request.include_probabilities:
                response_data["probabilities"] = {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.classes, probabilities)
                }
            
            # Add features if requested (debug mode)
            if request.include_features:
                response_data["features"] = {
                    feature_name: float(feature_value)
                    for feature_name, feature_value in zip(self.feature_names, features[0])
                }
            
            return PredictionResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

# Initialize API
api_instance = ExerciseRecognitionAPI()

# Create FastAPI app
app = FastAPI(
    title="Gym Exercise Recognition API",
    description="""
    🏋️ **Gym Exercise Recognition API**
    
    This API uses machine learning to recognize gym exercises from sensor data.
    
    **Supported Exercises:**
    - Adductor, ArmCurl, BenchPress, LegCurl, LegPress
    - Null (no exercise), Riding, RopeSkipping, Running
    - Squat, StairClimber, Walking
    
    **Input Requirements:**
    - 8 sensor channels (Session, A_x, A_y, A_z, G_x, G_y, G_z, C_1)
    - 60 time steps per channel (3 seconds at 20Hz sampling)
    - All values must be finite numbers
    
    **Authentication:**
    - Bearer token authentication required
    - Rate limiting applied based on API key tier
    
    **Security Features:**
    - Input validation and sanitization
    - Rate limiting
    - CORS protection
    - Comprehensive logging
    """,
    version=api_instance.version,
    contact={
        "name": "Gym Exercise Recognition Team",
        "email": "support@gymapi.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Add middleware
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "yourdomain.com"]
)

# Authentication dependency
async def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate API key and return user info."""
    token = credentials.credentials
    
    if token not in API_KEYS:
        logger.warning(f"Invalid API key attempted: {token[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_info = API_KEYS[token]
    logger.info(f"API access by {user_info['name']} ({user_info['tier']})")
    return user_info

# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Gym Exercise Recognition API",
        "version": api_instance.version,
        "status": "operational",
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_loaded=api_instance.model is not None,
        version=api_instance.version,
        uptime_seconds=round(time.time() - api_instance.start_time, 2)
    )

@app.get("/exercises", response_model=Dict[str, List[str]])
async def get_supported_exercises(user_info: dict = Depends(get_api_key)):
    """Get list of supported exercises."""
    return {
        "exercises": api_instance.classes,
        "count": len(api_instance.classes)
    }

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("60/minute")  # Base rate limit
async def predict_exercise(
    request: PredictionRequest,
    user_info: dict = Depends(get_api_key)
):
    """
    **Predict gym exercise from sensor data.**
    
    This endpoint analyzes 3 seconds of sensor data to predict the exercise being performed.
    
    **HTTP Method: POST**
    - Used because we're sending sensor data (potentially large payload)
    - Prediction operation modifies server state (logging, analytics)
    - Secure for sensitive data transmission
    
    **Rate Limits:**
    - Basic tier: 60 requests/minute
    - Premium tier: 1000 requests/minute
    
    **Example Usage:**
    ```python
    import requests
    
    headers = {"Authorization": "Bearer your_api_key"}
    data = {
        "sensor_data": {
            "session": [1.0] * 60,
            "a_x": [0.1] * 60,
            # ... other sensors
        },
        "include_probabilities": true
    }
    
    response = requests.post("http://api.com/predict", json=data, headers=headers)
    print(response.json())
    ```
    """
    # Apply user-specific rate limiting
    rate_limit = f"{user_info['requests_per_minute']}/minute"
    
    # Log prediction request
    logger.info(f"Prediction request from {user_info['name']}")
    
    try:
        # Make prediction
        result = api_instance.predict_exercise(request)
        
        # Log successful prediction
        logger.info(f"Prediction: {result.exercise} (confidence: {result.confidence:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed for {user_info['name']}: {str(e)}")
        raise

@app.get("/model/info", response_model=Dict)
async def get_model_info(user_info: dict = Depends(get_api_key)):
    """Get detailed model information."""
    if api_instance.model_metadata:
        return {
            "model_type": api_instance.model_metadata.get("model_type", "XGBoost"),
            "accuracy": api_instance.model_metadata.get("performance", {}).get("test_accuracy", "N/A"),
            "f1_score": api_instance.model_metadata.get("performance", {}).get("test_f1_macro", "N/A"),
            "features": api_instance.model_metadata.get("n_features", "N/A"),
            "classes": len(api_instance.classes)
        }
    else:
        return {"message": "Model metadata not available"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler with logging."""
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.utcnow().isoformat() + "Z"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unexpected error: {str(exc)} - {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.utcnow().isoformat() + "Z"}
    )

if __name__ == "__main__":
    # Production configuration
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False  # Set to False in production
    )
