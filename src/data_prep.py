"""
Downloads RecGym via ucimlrepo, standardizes sensors, builds windows (no label bleed),
and performs subject-aware splits. Outputs data/processed/windows.npz
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import GroupShuffleSplit
from ucimlrepo import fetch_ucirepo
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

def load_recgym() -> pd.DataFrame:
    ds = fetch_ucirepo(id=1128)  # RecGym id on UCI
    X = ds.data.features
    y = ds.data.targets
    return pd.concat([X, y], axis=1)

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
    print("Fetching RecGym from UCI...")
    df = load_recgym()
    label_col = _pick_one(df.columns.tolist(), LABEL_CANDIDATES)
    group_col = _pick_one(df.columns.tolist(), GROUP_CANDIDATES)
    sensors = [c for c in SENSOR_COLS if c in df.columns]
    if len(sensors) < 6:
        raise ValueError(f"Missing expected sensor columns. Found: {sensors}")

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

if __name__ == "__main__":
    main()
