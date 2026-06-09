from funcs import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, accuracy_score
)

###### DATA loading : 

def collect_all_files():
    from collections import defaultdict
    file_labels = defaultdict(set)
    for label in CHOSEN_LABELS:
        for fname in fetch_files_per_label(label, randomize=False):
            file_labels[fname].add(label)
    return list(file_labels.keys()), dict(file_labels)
 
 
def split(file_names, file_labels, test_size=0.2, val_size=0.1, seed=42):
    n = len(file_names)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
 
    n_test = int(n * test_size)
    n_val  = int(n * val_size)
 
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
 
    train_files = [file_names[i] for i in train_idx]
    val_files   = [file_names[i] for i in val_idx]
    test_files  = [file_names[i] for i in test_idx]
 
    _print_split_stats(train_files, val_files, test_files, file_labels)
    return train_files, val_files, test_files
 
 
def _print_split_stats(train_files, val_files, test_files, file_labels):
    total = len(train_files) + len(val_files) + len(test_files)
    print(f"\nSplit:  total={total}  train={len(train_files)} ({len(train_files)/total:.0%})  "
          f"val={len(val_files)} ({len(val_files)/total:.0%})  "
          f"test={len(test_files)} ({len(test_files)/total:.0%})")
    print(f"\n{'label':<18} {'train':>6} {'val':>6} {'test':>6}")
    print("-" * 38)
    for lbl in CHOSEN_LABELS:
        tr = sum(1 for f in train_files if lbl in file_labels[f])
        v  = sum(1 for f in val_files   if lbl in file_labels[f])
        te = sum(1 for f in test_files  if lbl in file_labels[f])
        print(f"  {lbl:<16} {tr:>6} {v:>6} {te:>6}")
 
 
def build_dataloaders(data_dir, tokenizer, domain,
                      test_size=0.2, val_size=0.1,
                      batch_size=32, num_workers=4,
                      max_length=512, seed=42):
 
    file_names, file_labels = collect_all_files()
    train_files, val_files, test_files = split(
        file_names, file_labels,
        test_size=test_size, val_size=val_size, seed=seed
    )
 
    train_ds = CodeforceDataset(data_dir, train_files, tokenizer, domain, max_length)
    val_ds   = CodeforceDataset(data_dir, val_files,   tokenizer, domain, max_length)
    test_ds  = CodeforceDataset(data_dir, test_files,  tokenizer, domain, max_length)
 
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
 
    return train_loader, val_loader, test_loader



######## Training functools : 

class MLPHead(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_labels, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_labels),
        )

    def forward(self, x):
        return self.net(x)   # logits


# ── Loss with pos_weight ──────────────────────────────────────────────────
def build_criterion(Y_train: np.ndarray, device):
    counts     = Y_train.sum(axis=0).astype(float)        # positives per label
    neg_counts = len(Y_train) - counts                     # negatives per label
    pos_weight = torch.tensor(neg_counts / np.clip(counts, 1, None),
                               dtype=torch.float32).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)




######### Metrics 


def compute_metrics(Y_true: np.ndarray, Y_pred_logits: np.ndarray, threshold=0.5):
    Y_pred = (torch.sigmoid(torch.tensor(Y_pred_logits)).numpy() >= threshold).astype(int)

    micro_f1   = f1_score(Y_true, Y_pred, average="micro",    zero_division=0)
    macro_f1   = f1_score(Y_true, Y_pred, average="macro",    zero_division=0)
    hloss      = hamming_loss(Y_true, Y_pred)
    subset_acc = accuracy_score(Y_true, Y_pred)

    # Manhattan distance between true and predicted binary vectors
    # = number of label mismatches per sample, averaged across dataset
    manhattan = np.abs(Y_true - Y_pred).sum(axis=1).mean()

    per_label = {}
    for j, lbl in enumerate(CHOSEN_LABELS):
        per_label[lbl] = {
            "f1":        f1_score(Y_true[:, j],      Y_pred[:, j], zero_division=0),
            "precision": precision_score(Y_true[:, j], Y_pred[:, j], zero_division=0),
            "recall":    recall_score(Y_true[:, j],   Y_pred[:, j], zero_division=0),
        }

    return {
        "micro_f1":    micro_f1,
        "macro_f1":    macro_f1,
        "hamming":     hloss,
        "subset_acc":  subset_acc,
        "manhattan":   manhattan,
        "per_label":   per_label,
    }


def print_metrics(metrics: dict, split="val"):
    print(f"\n── {split} metrics ───────────────────────────────────────────")
    print(f"  Micro  F1       : {metrics['micro_f1']:.4f}")
    print(f"  Macro  F1       : {metrics['macro_f1']:.4f}")
    print(f"  Hamming loss    : {metrics['hamming']:.4f}")
    print(f"  Subset accuracy : {metrics['subset_acc']:.4f}")
    print(f"  Manhattan dist  : {metrics['manhattan']:.4f}  (avg label mismatches/sample)")
    print(f"\n  {'label':<18} {'F1':>6} {'Precision':>10} {'Recall':>8}")
    print("  " + "-" * 46)
    for lbl, s in metrics["per_label"].items():
        print(f"  {lbl:<18} {s['f1']:>6.3f} {s['precision']:>10.3f} {s['recall']:>8.3f}")


# ── Threshold tuning on val ───────────────────────────────────────────────
def tune_thresholds(Y_val: np.ndarray, logits_val: np.ndarray, steps=20):
    probs = torch.sigmoid(torch.tensor(logits_val)).numpy()
    thresholds = np.linspace(0.1, 0.9, steps)
    best_t = np.full(len(CHOSEN_LABELS), 0.5)

    for j, lbl in enumerate(CHOSEN_LABELS):
        best_f1 = 0
        for t in thresholds:
            preds = (probs[:, j] >= t).astype(int)
            f1    = f1_score(Y_val[:, j], preds, zero_division=0)
            if f1 > best_f1:
                best_f1   = f1
                best_t[j] = t
        print(f"  {lbl:<18} threshold={best_t[j]:.2f}  F1={best_f1:.3f}")

    return best_t

# ── pos_weight from split stats ───────────────────────────────────────────
def compute_pos_weight(train_files, file_labels, device):
    """
    Compute pos_weight for BCEWithLogitsLoss directly from the split stats
    already available — no need to re-scan files or re-load labels.
 
    pos_weight[j] = n_negatives[j] / n_positives[j]
    """
    n_total = len(train_files)
    counts  = np.array([
        sum(1 for f in train_files if lbl in file_labels[f])
        for lbl in CHOSEN_LABELS
    ], dtype=np.float32)
 
    neg_counts = n_total - counts
    pos_weight = neg_counts / np.clip(counts, 1, None)
 
    print("\npos_weight per label (from split stats):")
    for lbl, pw, c in zip(CHOSEN_LABELS, pos_weight, counts):
        print(f"  {lbl:<18}  positives={int(c):>5}  pos_weight={pw:.2f}")
 
    return torch.tensor(pos_weight, dtype=torch.float32).to(device)

# ── collect logits + labels from a loader ─────────────────────────────────
def run_inference(model, loader, device, encoder=None):
    """
    Pass batches through optional frozen encoder then the head.
    Returns (all_logits, all_labels) as numpy arrays.
    """
    model.eval()
    all_logits, all_labels = [], []
 
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
 
            if encoder is not None:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs  = encoder(input_ids=input_ids, attention_mask=attention_mask)
                emb      = emb = mean_pooling(outputs.last_hidden_state, attention_mask)
                logits   = model(emb)
            else:
                # embeddings already precomputed and stored as input_ids
                logits = model(batch["input_ids"].to(device))
 
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
 
    return np.vstack(all_logits), np.vstack(all_labels)






################ Main Loops

def train(model, encoder,
          train_loader, val_loader, test_loader,
          train_files, file_labels,
          epochs=30, lr=1e-3, device="cpu"):
 
    device    = torch.device(device)
    model     = model.to(device)
    if encoder is not None:
        encoder = encoder.to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False # freezing the encoder, we don't want to train it.
 
    pos_weight = compute_pos_weight(train_files, file_labels, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
 
    best_macro_f1  = 0
    best_weights   = None
    best_logits_val = None
    best_labels_val = None
 
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
 
        for batch in train_loader:
            labels = batch["label"].to(device)
 
            if encoder is not None:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.no_grad():
                    outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                    emb     = mean_pooling(outputs.last_hidden_state, attention_mask)

                logits = model(emb)
            else:
                logits = model(batch["input_ids"].to(device))
 
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
 
        logits_val, labels_val = run_inference(model, val_loader, device, encoder)
        m = compute_metrics(labels_val, logits_val)
        scheduler.step(m["macro_f1"])
 
        print(f"Epoch {epoch:>3}  loss={total_loss/len(train_loader):.4f}"
              f"  macro_F1={m['macro_f1']:.4f}  micro_F1={m['micro_f1']:.4f}"
              f"  manhattan={m['manhattan']:.3f}")
 
        if m["macro_f1"] > best_macro_f1:
            best_macro_f1   = m["macro_f1"]
            best_weights    = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), "best_model.pt")
            best_logits_val = logits_val
            best_labels_val = labels_val
 
    model.load_state_dict(best_weights)
    print(f"\nBest val macro F1: {best_macro_f1:.4f}")
 
    thresholds = tune_thresholds(best_labels_val, best_logits_val)
    np.save("best_thresholds.npy", thresholds)
    
    return model, thresholds



def evaluate(model, encoder, test_loader, thresholds, device="cpu"):
    device = torch.device(device)
    model.to(device)
 
    logits_test, labels_test = run_inference(model, test_loader, device, encoder)
    probs = torch.sigmoid(torch.tensor(logits_test)).numpy()
    return probs,labels_test
 