import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

PROC = Path("data/processed/windows.npz")
OUT = Path("models"); OUT.mkdir(exist_ok=True)

def stats_features(X):
    # Reduce each window (T,C) to a small vector: mean/std/min/max per channel
    feats = [np.mean(X, 1), np.std(X, 1), np.min(X, 1), np.max(X, 1)]
    return np.concatenate(feats, axis=1)  # shape: (N, 4*C)

def main():
    d = np.load(PROC, allow_pickle=True)
    X_tr, y_tr = d["X_tr"], d["y_tr"]
    X_va, y_va = d["X_va"], d["y_va"]
    X_te, y_te = d["X_te"], d["y_te"]

    Xtr = stats_features(X_tr); Xva = stats_features(X_va); Xte = stats_features(X_te)

    le = LabelEncoder()
    ytr = le.fit_transform(y_tr); yva = le.transform(y_va); yte = le.transform(y_te)

    # Simple but strong baseline
    clf = XGBClassifier(
        n_estimators=600, max_depth=8, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=42
    )
    # train on train+val
    clf.fit(np.vstack([Xtr, Xva]), np.hstack([ytr, yva]))

    ypred = clf.predict(Xte)
    print(classification_report(yte, ypred, target_names=le.classes_))
    print("Macro-F1:", round(f1_score(yte, ypred, average="macro"), 3))

    joblib.dump(clf, OUT / "xgb_baseline.joblib")
    (OUT / "labels.json").write_text(json.dumps({"classes": le.classes_.tolist()}))
    print("Saved model to models/xgb_baseline.joblib")

if __name__ == "__main__":
    main()
