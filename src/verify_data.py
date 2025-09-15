"""
Data verification script to validate the processed gym exercises data.
Checks data integrity, shapes, distributions, and potential issues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the processed windows.npz file."""
    data_file = Path("data/processed/windows.npz")
    if not data_file.exists():
        raise FileNotFoundError(
            f"Processed data not found at {data_file}. "
            "Please run 'python src/data_prep.py' first."
        )
    
    data = np.load(data_file, allow_pickle=True)
    return data

def verify_data_shapes(data):
    """Verify the shapes and basic structure of the data."""
    print("=" * 60)
    print("📊 DATA SHAPES AND STRUCTURE")
    print("=" * 60)
    
    # Extract arrays
    X_tr, y_tr = data['X_tr'], data['y_tr']
    X_va, y_va = data['X_va'], data['y_va']
    X_te, y_te = data['X_te'], data['y_te']
    sensors = data['sensors']
    label_name = data['label_name']
    
    print(f"Training set:   X: {X_tr.shape}, y: {y_tr.shape}")
    print(f"Validation set: X: {X_va.shape}, y: {y_va.shape}")
    print(f"Test set:       X: {X_te.shape}, y: {y_te.shape}")
    print(f"Sensor columns: {sensors}")
    print(f"Label column:   {label_name[0]}")
    
    # Verify consistency
    assert X_tr.shape[0] == y_tr.shape[0], "Training X and y have different sample counts"
    assert X_va.shape[0] == y_va.shape[0], "Validation X and y have different sample counts"
    assert X_te.shape[0] == y_te.shape[0], "Test X and y have different sample counts"
    assert X_tr.shape[2] == len(sensors), "Feature dimension doesn't match sensor count"
    
    print("✅ All shapes are consistent!")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, sensors, label_name[0]

def verify_data_quality(X_tr, y_tr, X_va, y_va, X_te, y_te):
    """Check for data quality issues."""
    print("\n" + "=" * 60)
    print("🔍 DATA QUALITY CHECKS")
    print("=" * 60)
    
    # Check for NaN values
    nan_tr = np.isnan(X_tr).sum()
    nan_va = np.isnan(X_va).sum()
    nan_te = np.isnan(X_te).sum()
    
    print(f"NaN values - Train: {nan_tr}, Val: {nan_va}, Test: {nan_te}")
    
    # Check for infinite values
    inf_tr = np.isinf(X_tr).sum()
    inf_va = np.isinf(X_va).sum()
    inf_te = np.isinf(X_te).sum()
    
    print(f"Inf values - Train: {inf_tr}, Val: {inf_va}, Test: {inf_te}")
    
    # Check data ranges
    print(f"\nData ranges:")
    print(f"Train - Min: {X_tr.min():.3f}, Max: {X_tr.max():.3f}")
    print(f"Val   - Min: {X_va.min():.3f}, Max: {X_va.max():.3f}")
    print(f"Test  - Min: {X_te.min():.3f}, Max: {X_te.max():.3f}")
    
    # Check if data is standardized (should be roughly mean=0, std=1)
    print(f"\nStandardization check (should be ~0 mean, ~1 std):")
    print(f"Train - Mean: {X_tr.mean():.3f}, Std: {X_tr.std():.3f}")
    print(f"Val   - Mean: {X_va.mean():.3f}, Std: {X_va.std():.3f}")
    print(f"Test  - Mean: {X_te.mean():.3f}, Std: {X_te.std():.3f}")
    
    if nan_tr + nan_va + nan_te == 0 and inf_tr + inf_va + inf_te == 0:
        print("✅ No NaN or infinite values found!")
    else:
        print("❌ Found NaN or infinite values - check data preprocessing!")

def verify_label_distribution(y_tr, y_va, y_te):
    """Check label distributions across splits."""
    print("\n" + "=" * 60)
    print("🏷️  LABEL DISTRIBUTION")
    print("=" * 60)
    
    # Count labels in each split
    tr_counts = Counter(y_tr)
    va_counts = Counter(y_va)
    te_counts = Counter(y_te)
    
    # Get all unique labels
    all_labels = sorted(set(list(y_tr) + list(y_va) + list(y_te)))
    
    print(f"{'Label':<15} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 55)
    
    for label in all_labels:
        tr_count = tr_counts.get(label, 0)
        va_count = va_counts.get(label, 0)
        te_count = te_counts.get(label, 0)
        total = tr_count + va_count + te_count
        print(f"{str(label):<15} {tr_count:<8} {va_count:<8} {te_count:<8} {total:<8}")
    
    print("-" * 55)
    print(f"{'TOTAL':<15} {len(y_tr):<8} {len(y_va):<8} {len(y_te):<8} {len(y_tr)+len(y_va)+len(y_te):<8}")
    
    # Check for missing labels in any split
    missing_in_val = set(y_tr) - set(y_va) if len(y_va) > 0 else set()
    missing_in_test = set(y_tr) - set(y_te) if len(y_te) > 0 else set()
    
    if missing_in_val:
        print(f"⚠️  Labels missing in validation: {missing_in_val}")
    if missing_in_test:
        print(f"⚠️  Labels missing in test: {missing_in_test}")
    
    if not missing_in_val and not missing_in_test:
        print("✅ All labels present in all splits!")

def verify_split_ratios(X_tr, X_va, X_te):
    """Verify the split ratios are approximately correct."""
    print("\n" + "=" * 60)
    print("📊 SPLIT RATIOS")
    print("=" * 60)
    
    total_samples = len(X_tr) + len(X_va) + len(X_te)
    tr_ratio = len(X_tr) / total_samples
    va_ratio = len(X_va) / total_samples
    te_ratio = len(X_te) / total_samples
    
    print(f"Total samples: {total_samples}")
    print(f"Train: {len(X_tr):6} ({tr_ratio:.1%})")
    print(f"Val:   {len(X_va):6} ({va_ratio:.1%})")
    print(f"Test:  {len(X_te):6} ({te_ratio:.1%})")
    
    # Expected ratios: 70% train, 15% val, 15% test
    expected_ratios = [0.70, 0.15, 0.15]
    actual_ratios = [tr_ratio, va_ratio, te_ratio]
    
    print(f"\nExpected vs Actual:")
    split_names = ["Train", "Val", "Test"]
    for name, expected, actual in zip(split_names, expected_ratios, actual_ratios):
        diff = abs(expected - actual)
        status = "✅" if diff < 0.05 else "⚠️"
        print(f"{name}: {expected:.1%} expected, {actual:.1%} actual {status}")

def create_visualization_plots(X_tr, y_tr, sensors):
    """Create visualization plots to inspect the data."""
    print("\n" + "=" * 60)
    print("📈 CREATING VISUALIZATION PLOTS")
    print("=" * 60)
    
    # Create output directory
    viz_dir = Path("data/processed/verification_plots")
    viz_dir.mkdir(exist_ok=True)
    
    # Plot 1: Sensor data distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, sensor in enumerate(sensors[:6]):  # Plot first 6 sensors
        if i < len(axes):
            # Flatten the sensor data across all windows and samples
            sensor_data = X_tr[:, :, i].flatten()
            axes[i].hist(sensor_data, bins=50, alpha=0.7)
            axes[i].set_title(f'{sensor} Distribution')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'sensor_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Label distribution
    plt.figure(figsize=(12, 6))
    label_counts = Counter(y_tr)
    labels, counts = zip(*sorted(label_counts.items()))
    
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.title('Training Set Label Distribution')
    plt.xlabel('Exercise Type')
    plt.ylabel('Number of Windows')
    plt.tight_layout()
    plt.savefig(viz_dir / 'label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Sample time series for each sensor
    fig, axes = plt.subplots(len(sensors), 1, figsize=(12, 2*len(sensors)))
    if len(sensors) == 1:
        axes = [axes]
    
    # Take first window of first sample
    sample_window = X_tr[0]  # Shape: (window_size, n_sensors)
    
    for i, sensor in enumerate(sensors):
        axes[i].plot(sample_window[:, i])
        axes[i].set_title(f'{sensor} - Sample Time Series')
        axes[i].set_ylabel('Value')
        if i == len(sensors) - 1:
            axes[i].set_xlabel('Time Step')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'sample_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to: {viz_dir}")
    print("   - sensor_distributions.png: Distribution of sensor values")
    print("   - label_distribution.png: Distribution of exercise labels")
    print("   - sample_time_series.png: Sample time series for each sensor")

def main():
    """Main verification function."""
    print("🔍 VERIFYING PROCESSED GYM EXERCISES DATA")
    print("=" * 60)
    
    try:
        # Load processed data
        data = load_processed_data()
        
        # Verify shapes and structure
        X_tr, y_tr, X_va, y_va, X_te, y_te, sensors, label_name = verify_data_shapes(data)
        
        # Verify data quality
        verify_data_quality(X_tr, y_tr, X_va, y_va, X_te, y_te)
        
        # Verify label distribution
        verify_label_distribution(y_tr, y_va, y_te)
        
        # Verify split ratios
        verify_split_ratios(X_tr, X_va, X_te)
        
        # Create visualizations
        create_visualization_plots(X_tr, y_tr, sensors)
        
        print("\n" + "=" * 60)
        print("✅ DATA VERIFICATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Your processed data looks good and ready for model training!")
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        print("Please check your data preparation pipeline.")

if __name__ == "__main__":
    main()
