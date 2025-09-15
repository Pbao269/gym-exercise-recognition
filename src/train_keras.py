"""
Deep Learning (Keras/TensorFlow) training for gym exercises recognition.
Uses CNN-LSTM architecture for time series classification.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

PROC = Path("data/processed/windows.npz")
OUT = Path("models"); OUT.mkdir(exist_ok=True)

def build_improved_model(seq_len, n_ch, n_classes):
    """
    Build an improved CNN-LSTM model optimized for gym exercise recognition.
    
    Architecture:
    - Multiple CNN layers for feature extraction
    - Bidirectional LSTM for temporal modeling
    - Attention mechanism for important time steps
    - Dropout and regularization for generalization
    """
    inputs = keras.Input(shape=(seq_len, n_ch), name='sensor_input')
    
    # CNN Feature Extraction Blocks
    # Block 1: Initial feature extraction
    x = layers.Conv1D(64, 7, padding="same", activation="relu", name='conv1d_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.MaxPool1D(2, name='pool_1')(x)
    x = layers.Dropout(0.1, name='dropout_1')(x)
    
    # Block 2: Deeper feature extraction
    x = layers.Conv1D(128, 5, padding="same", activation="relu", name='conv1d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.MaxPool1D(2, name='pool_2')(x)
    x = layers.Dropout(0.1, name='dropout_2')(x)
    
    # Block 3: High-level features
    x = layers.Conv1D(256, 3, padding="same", activation="relu", name='conv1d_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Dropout(0.2, name='dropout_3')(x)
    
    # Bidirectional LSTM for temporal modeling
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='bilstm_1'
    )(x)
    
    # Attention mechanism (simplified)
    attention = layers.Dense(1, activation='tanh', name='attention_weights')(x)
    attention = layers.Softmax(axis=1, name='attention_softmax')(attention)
    x = layers.Multiply(name='attention_multiply')([x, attention])
    
    # Final LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3),
        name='bilstm_2'
    )(x)
    
    # Classification head
    x = layers.Dense(128, activation="relu", name='dense_1')(x)
    x = layers.Dropout(0.4, name='dropout_final')(x)
    outputs = layers.Dense(n_classes, activation="softmax", name='predictions')(x)
    
    model = keras.Model(inputs, outputs, name='GymExerciseClassifier')
    
    # Use a more sophisticated optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", "sparse_top_k_categorical_accuracy"]
    )
    
    return model

def plot_training_history(history, save_path):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Deep Learning Gym Exercise Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("🧠 Training Deep Learning Model for Gym Exercise Recognition")
    print("=" * 70)
    
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
    
    # Prepare data
    seq_len, n_ch = X_tr.shape[1], X_tr.shape[2]
    le = LabelEncoder()
    ytr = le.fit_transform(y_tr)
    yva = le.transform(y_va)
    yte = le.transform(y_te)
    n_classes = len(le.classes_)
    
    print(f"\nData preparation completed:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of channels: {n_ch}")
    print(f"  Number of classes: {n_classes}")
    print(f"  Classes: {list(le.classes_)}")
    
    # Build improved model
    print("\nBuilding improved CNN-LSTM model...")
    model = build_improved_model(seq_len, n_ch, n_classes)
    
    print(f"Model built successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Setup advanced callbacks
    print("\nSetting up training callbacks...")
    callbacks_list = [
        # Learning rate reduction
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=7,
            factor=0.5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=str(OUT / 'best_model_checkpoint.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    print("\n" + "="*70)
    print("TRAINING DEEP LEARNING MODEL")
    print("="*70)
    
    print("Starting training with advanced callbacks...")
    print("- Early stopping: 15 epochs patience")
    print("- Learning rate reduction: 7 epochs patience")
    print("- Model checkpointing: Save best validation accuracy")
    
    history = model.fit(
        X_tr, ytr,
        validation_data=(X_va, yva),
        epochs=100,  # More epochs with early stopping
        batch_size=64,  # Smaller batch size for better gradients
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Plot training history
    history_plot_path = OUT / "training_history_keras.png"
    plot_training_history(history, history_plot_path)
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("VALIDATION SET EVALUATION")
    print("="*70)
    
    yva_pred_probs = model.predict(X_va, verbose=0)
    yva_pred = np.argmax(yva_pred_probs, axis=1)
    
    val_accuracy = accuracy_score(yva, yva_pred)
    val_f1_macro = f1_score(yva, yva_pred, average='macro')
    val_f1_weighted = f1_score(yva, yva_pred, average='weighted')
    
    print(f"Validation Accuracy: {val_accuracy:.3f}")
    print(f"Validation F1 (Macro): {val_f1_macro:.3f}")
    print(f"Validation F1 (Weighted): {val_f1_weighted:.3f}")
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    yte_pred_probs = model.predict(X_te, verbose=0)
    yte_pred = np.argmax(yte_pred_probs, axis=1)
    
    test_accuracy = accuracy_score(yte, yte_pred)
    test_f1_macro = f1_score(yte, yte_pred, average='macro')
    test_f1_weighted = f1_score(yte, yte_pred, average='weighted')
    
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test F1 (Macro): {test_f1_macro:.3f}")
    print(f"Test F1 (Weighted): {test_f1_weighted:.3f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(yte, yte_pred, target_names=le.classes_))
    
    # Save confusion matrix
    cm_path = OUT / "confusion_matrix_keras.png"
    plot_confusion_matrix(yte, yte_pred, le.classes_, cm_path)
    
    # Training summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    final_epoch = len(history.history['loss'])
    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    
    print(f"Training completed in {final_epoch} epochs")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Best validation loss: {best_val_loss:.3f}")
    print(f"Final test accuracy: {test_accuracy:.3f}")
    
    # Save model and metadata
    print("\n" + "="*70)
    print("SAVING MODEL AND RESULTS")
    print("="*70)
    
    # Save the final model
    model_path = OUT / "keras_gym_classifier.keras"
    model.save(model_path)
    
    # Save comprehensive metadata
    metadata = {
        "model_type": "CNN-LSTM Deep Learning",
        "classes": le.classes_.tolist(),
        "n_classes": len(le.classes_),
        "sequence_length": seq_len,
        "n_channels": n_ch,
        "sensors": sensors.tolist(),
        "total_parameters": int(model.count_params()),
        "training": {
            "epochs_completed": final_epoch,
            "batch_size": 64,
            "best_val_accuracy": float(best_val_acc),
            "best_val_loss": float(best_val_loss)
        },
        "performance": {
            "test_accuracy": float(test_accuracy),
            "test_f1_macro": float(test_f1_macro),
            "test_f1_weighted": float(test_f1_weighted),
            "val_accuracy": float(val_accuracy),
            "val_f1_macro": float(val_f1_macro),
            "val_f1_weighted": float(val_f1_weighted)
        }
    }
    
    metadata_path = OUT / "keras_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Also save labels for compatibility
    labels_path = OUT / "keras_labels.json"
    with open(labels_path, 'w') as f:
        json.dump({"classes": le.classes_.tolist()}, f)
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Checkpoint saved: {OUT / 'best_model_checkpoint.keras'}")
    print(f"✅ Metadata saved: {metadata_path}")
    print(f"✅ Labels saved: {labels_path}")
    print(f"✅ Training history: {history_plot_path}")
    print(f"✅ Confusion matrix: {cm_path}")
    
    print(f"\n🎯 Final Results Summary:")
    print(f"   Test Accuracy: {test_accuracy:.1%}")
    print(f"   Test F1-Score: {test_f1_macro:.3f}")
    print(f"   Model Parameters: {model.count_params():,}")
    print(f"   Training Time: {final_epoch} epochs")
    print(f"   Ready for deployment! 🚀")

if __name__ == "__main__":
    main()
