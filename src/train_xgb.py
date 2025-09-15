"""
XGBoost training for gym exercises recognition.
Uses statistical features extracted from time series windows.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

PROC = Path("data/processed/windows.npz")
OUT = Path("models"); OUT.mkdir(exist_ok=True)

def extract_statistical_features(X):
    """
    Extract comprehensive statistical features from time series windows.
    
    Args:
        X: Shape (N, T, C) where N=samples, T=time_steps, C=channels
        
    Returns:
        Features: Shape (N, features_per_channel * C)
    """
    # Basic statistical features
    mean_feats = np.mean(X, axis=1)      # (N, C)
    std_feats = np.std(X, axis=1)        # (N, C)
    min_feats = np.min(X, axis=1)        # (N, C)
    max_feats = np.max(X, axis=1)        # (N, C)
    
    # Additional features for better gym exercise recognition
    median_feats = np.median(X, axis=1)  # (N, C)
    q25_feats = np.percentile(X, 25, axis=1)  # (N, C)
    q75_feats = np.percentile(X, 75, axis=1)  # (N, C)
    range_feats = max_feats - min_feats  # (N, C) - Range of motion
    
    # Energy and activity features
    rms_feats = np.sqrt(np.mean(X**2, axis=1))  # RMS energy
    var_feats = np.var(X, axis=1)       # Variance
    
    # Combine all features
    features = [
        mean_feats, std_feats, min_feats, max_feats,
        median_feats, q25_feats, q75_feats, range_feats,
        rms_feats, var_feats
    ]
    
    return np.concatenate(features, axis=1)  # Shape: (N, 10*C)

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - XGBoost Gym Exercise Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("🏋️ Training XGBoost for Gym Exercise Recognition")
    print("=" * 60)
    
    # Load processed data
    print("Loading processed data...")
    d = np.load(PROC, allow_pickle=True)
    X_tr, y_tr = d["X_tr"], d["y_tr"]
    X_va, y_va = d["X_va"], d["y_va"]
    X_te, y_te = d["X_te"], d["y_te"]
    sensors = d["sensors"]
    
    print(f"Data loaded: Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}")
    print(f"Window shape: {X_tr.shape[1:]} (time_steps × sensors)")
    print(f"Sensors: {list(sensors)}")
    
    # Extract statistical features
    print("\nExtracting statistical features...")
    Xtr = extract_statistical_features(X_tr)
    Xva = extract_statistical_features(X_va)
    Xte = extract_statistical_features(X_te)
    
    print(f"Feature extraction completed:")
    print(f"  Original shape: {X_tr.shape} → Feature shape: {Xtr.shape}")
    print(f"  Features per sensor: {Xtr.shape[1] // len(sensors)}")
    
    # Encode labels
    le = LabelEncoder()
    ytr = le.fit_transform(y_tr)
    yva = le.transform(y_va)
    yte = le.transform(y_te)
    
    print(f"\nLabel encoding completed:")
    print(f"  Classes: {len(le.classes_)} - {list(le.classes_)}")
    
    # Configure XGBoost for multiclass gym exercise classification
    print("\nTraining XGBoost classifier...")
    clf = XGBClassifier(
        # Core parameters optimized for gym exercises
        n_estimators=800,           # More trees for better performance
        max_depth=10,               # Deeper trees for complex patterns
        learning_rate=0.03,         # Lower learning rate with more estimators
        
        # Regularization to prevent overfitting
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        
        # Sampling for robustness
        subsample=0.8,              # Row sampling
        colsample_bytree=0.8,       # Column sampling per tree
        colsample_bylevel=0.8,      # Column sampling per level
        
        # Performance and reproducibility
        tree_method="hist",         # Fast histogram-based method
        random_state=42,
        n_jobs=-1,                  # Use all CPU cores
        
        # Multiclass settings
        objective='multi:softprob', # Multiclass probability
        eval_metric='mlogloss'      # Multiclass log loss
    )
    
    # Train the classifier (simple training for compatibility)
    print("Training XGBoost classifier...")
    clf.fit(Xtr, ytr)
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)
    yva_pred = clf.predict(Xva)
    val_accuracy = accuracy_score(yva, yva_pred)
    val_f1_macro = f1_score(yva, yva_pred, average='macro')
    val_f1_weighted = f1_score(yva, yva_pred, average='weighted')
    
    print(f"Validation Accuracy: {val_accuracy:.3f}")
    print(f"Validation F1 (Macro): {val_f1_macro:.3f}")
    print(f"Validation F1 (Weighted): {val_f1_weighted:.3f}")
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    yte_pred = clf.predict(Xte)
    test_accuracy = accuracy_score(yte, yte_pred)
    test_f1_macro = f1_score(yte, yte_pred, average='macro')
    test_f1_weighted = f1_score(yte, yte_pred, average='weighted')
    
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test F1 (Macro): {test_f1_macro:.3f}")
    print(f"Test F1 (Weighted): {test_f1_weighted:.3f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(yte, yte_pred, target_names=le.classes_))
    
    # Feature importance analysis
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importance
    importance = clf.feature_importances_
    
    # Create feature names
    feature_types = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'range', 'rms', 'var']
    feature_names = []
    for sensor in sensors:
        for feat_type in feature_types:
            feature_names.append(f"{sensor}_{feat_type}")
    
    # Show top features
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("Top 15 Most Important Features:")
    for i, (_, row) in enumerate(feature_df.head(15).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<20} {row['importance']:.4f}")
    
    # Save confusion matrix
    cm_path = OUT / "confusion_matrix_xgb.png"
    plot_confusion_matrix(yte, yte_pred, le.classes_, cm_path)
    
    # Save model and metadata
    print("\n" + "="*60)
    print("SAVING MODEL AND RESULTS")
    print("="*60)
    
    model_path = OUT / "xgb_gym_classifier.joblib"
    joblib.dump(clf, model_path)
    
    # Save comprehensive metadata
    metadata = {
        "model_type": "XGBoost",
        "classes": le.classes_.tolist(),
        "n_classes": len(le.classes_),
        "n_features": Xtr.shape[1],
        "n_sensors": len(sensors),
        "sensors": sensors.tolist(),
        "feature_types": feature_types,
        "performance": {
            "test_accuracy": float(test_accuracy),
            "test_f1_macro": float(test_f1_macro),
            "test_f1_weighted": float(test_f1_weighted),
            "val_accuracy": float(val_accuracy),
            "val_f1_macro": float(val_f1_macro),
            "val_f1_weighted": float(val_f1_weighted)
        },
        "hyperparameters": clf.get_params()
    }
    
    metadata_path = OUT / "xgb_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature importance
    feature_df.to_csv(OUT / "feature_importance_xgb.csv", index=False)
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Metadata saved: {metadata_path}")
    print(f"✅ Confusion matrix: {cm_path}")
    print(f"✅ Feature importance: {OUT / 'feature_importance_xgb.csv'}")
    
    print(f"\n🎯 Final Results Summary:")
    print(f"   Test Accuracy: {test_accuracy:.1%}")
    print(f"   Test F1-Score: {test_f1_macro:.3f}")
    print(f"   Ready for deployment! 🚀")

if __name__ == "__main__":
    main()
