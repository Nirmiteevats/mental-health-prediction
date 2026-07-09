import os
import sys
import numpy as np
import pytest

# Add the parent directory to the path so we can import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import load_artifacts

def test_models_exist():
    """Test that all required model files exist in the models directory."""
    assert os.path.exists("models/mental_health_model.pkl"), "Model file missing"
    assert os.path.exists("models/le_gender.pkl"), "Gender encoder missing"
    assert os.path.exists("models/le_occupation.pkl"), "Occupation encoder missing"
    assert os.path.exists("models/le_target.pkl"), "Target encoder missing"

def test_load_artifacts():
    """Test that the artifacts load correctly and are of expected types."""
    model, le_gender, le_occupation, le_target = load_artifacts()
    
    assert model is not None, "Failed to load model"
    assert le_gender is not None, "Failed to load gender encoder"
    assert le_occupation is not None, "Failed to load occupation encoder"
    assert le_target is not None, "Failed to load target encoder"
    
    # Check that encoders have classes
    assert hasattr(le_gender, "classes_")
    assert hasattr(le_occupation, "classes_")
    assert hasattr(le_target, "classes_")

def test_prediction_shape():
    """Test that the model can make a prediction on dummy data."""
    model, _, _, _ = load_artifacts()
    
    # 15 features expected
    dummy_features = np.zeros((1, 15))
    
    prediction = model.predict(dummy_features)
    assert len(prediction) == 1, "Prediction should return exactly one item for a single row"
