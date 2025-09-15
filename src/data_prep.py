"""
Downloads RecGym via ucimlrepo, standardizes sensors, builds windows (no label bleed),
and performs subject-aware splits. Outputs data/processed/windows.npz
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import GroupShuffleSplit
# from ucimlrepo import fetch_ucirepo  # Not needed for manual download
from config import (WINDOW_SIZE, WINDOW_STRIDE, SENSOR_COLS,
                    LABEL_CANDIDATES, GROUP_CANDIDATES, RANDOM_STATE)

DATA_DIR = Path("data")
PROC_DIR = DATA_DIR / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def _pick_one(cols: List[str], candidates: List[str]) -> str:
    for c in candidates:
        if c in cols: return c
    low = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in low:
            return cols[low.index(cand.lower())]
    raise ValueError(f"Could not find any of {candidates} in columns: {cols[:15]}")

def load_gym_dataset() -> pd.DataFrame:
    """
    Load gym exercises dataset from manually downloaded files.
    Supports both RecGym and Kaggle gym datasets.
    """
    raw_dir = DATA_DIR / "raw"
    
    # Look for common file patterns
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Please download a gym dataset:\n"
            "Option 1 (Kaggle): https://www.kaggle.com/datasets/zhaxidelebsz/10-gym-exercises-with-615-abstracted-features\n"
            "Option 2 (UCI): https://uci-ics-mlr-prod.aws.uci.edu/dataset/1128/recgym:+gym+workouts+recognition+dataset+with+imu+and+capacitive+sensor-7\n"
            "Extract files to data/raw/"
        )
    
    # Load the first CSV file (or combine multiple if needed)
    if len(csv_files) == 1:
        print(f"Loading gym data from: {csv_files[0]}")
        df = pd.read_csv(csv_files[0])
    else:
        print(f"Found {len(csv_files)} CSV files, combining them...")
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def standardize(df: pd.DataFrame, sensors: List[str]) -> pd.DataFrame:
    # z-score channel-wise (mean=0, std=1) to help training stability
    for c in sensors:
        mu, sd = df[c].mean(), df[c].std()
        df[c] = 0.0 if (sd is None or sd == 0 or np.isnan(sd)) else (df[c] - mu) / sd
    return df

def window_consistent(df: pd.DataFrame, label_col: str, sensors: List[str]):
    """
    Slide fixed windows. Only keep windows where the label stays the same
    for the entire window (avoids mixing two activities).
    """
    Xw, yw = [], []
    vals = df[sensors].values
    labs = df[label_col].values
    n = len(df); i = 0
    while i + WINDOW_SIZE <= n:
        w_labs = labs[i:i+WINDOW_SIZE]
        if np.all(w_labs == w_labs[0]):
            Xw.append(vals[i:i+WINDOW_SIZE])
            yw.append(w_labs[0])
        i += WINDOW_STRIDE
    return np.array(Xw), np.array(yw)

def main():
    print("Loading gym exercises dataset...")
    df = load_gym_dataset()
    label_col = _pick_one(df.columns.tolist(), LABEL_CANDIDATES)
    
    # Try to find group column, but make it optional for some datasets
    try:
        group_col = _pick_one(df.columns.tolist(), GROUP_CANDIDATES)
    except ValueError:
        print("Warning: No group/subject column found. Will use row indices for grouping.")
        df['_subject_id'] = df.index // 1000  # Create artificial subjects
        group_col = '_subject_id'
    
    # Auto-detect sensor columns (numeric columns that aren't labels/groups)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sensors = [c for c in numeric_cols if c not in [label_col, group_col]]
    
    if len(sensors) < 3:
        raise ValueError(f"Need at least 3 sensor columns. Found: {sensors}")
    
    print(f"Using sensor columns: {sensors[:10]}{'...' if len(sensors) > 10 else ''}")
    print(f"Label column: {label_col}")
    print(f"Group column: {group_col}")

    # Clean & standardize
    df = df.dropna(subset=sensors + [label_col, group_col]).reset_index(drop=True)
    df = standardize(df, sensors)

    # Build windows per subject to keep time order within each person
    allX, ally, allg = [], [], []
    for g, gdf in df.groupby(group_col):
        Xw, yw = window_consistent(gdf, label_col, sensors)
        if len(yw):
            allX.append(Xw)
            ally.append(yw)
            allg.append(np.full_like(yw, g, dtype=object))

    X = np.concatenate(allX, axis=0)
    y = np.concatenate(ally, axis=0)
    groups = np.concatenate(allg, axis=0)
    print("Windows:", X.shape, "Labels:", y.shape, "Groups:", groups.shape)

    # Subject-aware splits: 70% train, 15% val, 15% test by subject (no leakage)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    tr, hold = next(gss1.split(X, y, groups))
    X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
    X_hold, y_hold, g_hold = X[hold], y[hold], groups[hold]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    va, te = next(gss2.split(X_hold, y_hold, g_hold))
    X_va, y_va = X_hold[va], y_hold[va]
    X_te, y_te = X_hold[te], y_hold[te]

    np.savez_compressed(PROC_DIR / "windows.npz",
                        X_tr=X_tr, y_tr=y_tr,
                        X_va=X_va, y_va=y_va,
                        X_te=X_te, y_te=y_te,
                        sensors=np.array(sensors), label_name=np.array([label_col]))
    print("Saved ->", PROC_DIR / "windows.npz")
    
    # Quick verification summary
    print("\n" + "="*50)
    print("QUICK VERIFICATION SUMMARY")
    print("="*50)
    print(f"Total windows: {len(X)}")
    print(f"Window shape: {X.shape[1:]} (time_steps × sensors)")
    print(f"Unique labels: {len(np.unique(y))}")
    print(f"Labels: {sorted(np.unique(y))}")
    print(f"Split sizes: Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}")
    print("Data processing completed successfully!")
    print("\nRun 'python src/verify_data.py' for detailed verification")

if __name__ == "__main__":
    main()
