# predict_tfidf.py

import json
import joblib
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import (f1_score, hamming_loss,
                              accuracy_score, classification_report,
                              average_precision_score)

# ── HARDCODED PATHS ───────────────────────────────────────────────────────────

FOLDER_PATH     = "best_model_All"
MODEL_PATH      = FOLDER_PATH+"/"+"model.pkl"
VECTORIZER_PATH = FOLDER_PATH+"/"+"vectorizer.pkl"
THRESHOLDS_PATH = FOLDER_PATH+"/"+"thresholds.json"
TEXT_FIELD      = "prob_desc_description"

# ── LOAD SHARED RESOURCES ─────────────────────────────────────────────────────

model      = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

with open(THRESHOLDS_PATH, "r") as f:
    thresholds = json.load(f)          # {label_name: threshold}

label_names = list(thresholds.keys())
thr_arr     = np.array(list(thresholds.values()))   # (n_labels,)


# ── LOAD RECORDS ──────────────────────────────────────────────────────────────

def load_records(input_path: str) -> tuple:
    p = Path(input_path)

    if p.is_dir():
        files = sorted(p.glob("*.json"))
    elif p.is_file():
        files = [p]
    else:
        raise ValueError(f"Input must be a .json file or a folder: {input_path}")

    texts, tags_list, file_names = [], [], []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            record = json.load(f)
        texts.append(record.get(TEXT_FIELD, "") or "")
        tags_list.append(record.get("tags", []))
        file_names.append(fp.name)

    print(f"Loaded {len(texts)} record(s)")
    return texts, tags_list, file_names


# ── PREDICT ───────────────────────────────────────────────────────────────────

def predict(texts: list) -> tuple:
    X      = vectorizer.transform(texts)
    Y_prob = model.predict_proba(X)
    Y_pred = (Y_prob >= thr_arr).astype(int)
    return Y_prob, Y_pred


# ── PRINT SINGLE ──────────────────────────────────────────────────────────────

def print_single(file_name, true_tags, Y_prob, Y_pred):
    predicted = [label_names[i] for i, p in enumerate(Y_pred[0]) if p == 1]

    print(f"\n── Prediction ───────────────────────────────────────────")
    print(f"  File            : {file_name}")
    print(f"  True tags       : {true_tags}")
    print(f"  Predicted tags  : {predicted}")
    print(f"\n  Probabilities (sorted):")
    for label, prob, thr in sorted(
            zip(label_names, Y_prob[0], thr_arr), key=lambda x: -x[1]):
        bar  = '█' * int(prob * 20)
        flag = " ✓" if prob >= thr else ""
        print(f"    {label:<22} {bar:<20} {prob:.4f}  (thr={thr:.2f}){flag}")


# ── PRINT METRICS ─────────────────────────────────────────────────────────────

def print_metrics(Y_true, Y_prob, Y_pred):

    print(f"\n── Metrics ──────────────────────────────────────────────")
    print(f"  Samples        : {len(Y_true)}")
    print(f"  Macro  F1      : {f1_score(Y_true, Y_pred, average='macro',    zero_division=0):.4f}")
    print(f"  Micro  F1      : {f1_score(Y_true, Y_pred, average='micro',    zero_division=0):.4f}")
    print(f"  Hamming Score   : {1-hamming_loss(Y_true, Y_pred):.4f}")
    print(f"  Subset Accuracy: {accuracy_score(Y_true, Y_pred):.4f}")
    print("\n" + classification_report(
        Y_true, Y_pred, target_names=label_names, zero_division=0
    ))


# ── BUILD Y_TRUE ──────────────────────────────────────────────────────────────

def build_y_true(tags_list):
    tag2id = {name: i for i, name in enumerate(label_names)}
    Y      = np.zeros((len(tags_list), len(label_names)))
    for i, tags in enumerate(tags_list):
        for tag in tags:
            if tag in tag2id:
                Y[i, tag2id[tag]] = 1
    return Y


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_tfidf.py <file.json | folder/>")
        sys.exit(1)

    texts, tags_list, file_names = load_records(sys.argv[1])
    Y_prob, Y_pred               = predict(texts)

    if len(texts) == 1:
        print_single(file_names[0], tags_list[0], Y_prob, Y_pred)
    else:
        Y_true = build_y_true(tags_list)
        print_metrics(Y_true, Y_prob, Y_pred)