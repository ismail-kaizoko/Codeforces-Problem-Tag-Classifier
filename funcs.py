import re
import json
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import html
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

import plotly.express as px
from collections import defaultdict
import os



CHOSEN_LABELS = [
    "math", "graphs", "strings", "number theory",
    "trees", "geometry", "games", "probabilities"
]

DATA_DIR         = Path(__file__).parent.parent / "code_classification_dataset"

OUT_FILE = "label_index.json"


label_to_files = defaultdict(set)  # tag -> set of filenames
others = set()                     # files with no chosen tag at all

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(DATA_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)

    tags = rec.get("tags", [])
    matched = [t for t in tags if t in CHOSEN_LABELS]

    if matched:
        for tag in matched:
            label_to_files[tag].add(fname)
    else:
        others.add(fname)

# Build final structure
index = {}
for label in CHOSEN_LABELS:
    files = sorted(label_to_files[label])
    index[label] = {"files": files, "count": len(files)}

index["others"] = {"files": sorted(others), "count": len(others)}

with open(OUT_FILE, "w") as f:
    json.dump(index, f, indent=2)

print(f"Saved → {OUT_FILE}\n")
for label, info in index.items():
    print(f"  {label:<20} {info['count']} files")


LABEL_INDEX_PATH = Path(__file__).parent / OUT_FILE




# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess_code(code: str) -> str:
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    code = code.replace('\t', '    ')
    code = re.sub(r'#[^\n]*', '', code)
    code = '\n'.join(line.rstrip() for line in code.split('\n'))
    code = re.sub(r'\n{3,}', '\n\n', code)
    return code.strip()

def decode_html_entities(text: str) -> str:
    """
    Step 0 — decode HTML entities before any LaTeX processing.
    Must run FIRST — &lt; must become < before LaTeX patterns fire.
    """
    return html.unescape(text)

def preprocess_text(text: str) -> str:
    if not text or str(text).strip() == "None":
        return "empty"

    # strip LaTeX delimiters — longest first to avoid partial matches
    text = decode_html_entities(text)
    text = re.sub(r'\$\$\$(.+?)\$\$\$', r' \1 ', text, flags=re.DOTALL)
    text = re.sub(r'\$\$(.+?)\$\$',     r' \1 ', text, flags=re.DOTALL)
    text = re.sub(r'\$(.+?)\$',         r' \1 ', text, flags=re.DOTALL)

    replacements = [
        (r'\\leq\b',                         ' less than or equal to '),
        (r'\\geq\b',                         ' greater than or equal to '),
        (r'\\le\b',                          ' less than or equal to '),
        (r'\\ge\b',                          ' greater than or equal to '),
        (r'\\neq\b',                         ' not equal to '),
        (r'\\ne\b',                          ' not equal to '),
        (r'\\equiv\b',                       ' equivalent to '),
        (r'\\approx\b',                      ' approximately '),
        (r'\\cdot\b',                        ' times '),
        (r'\\times\b',                       ' times '),
        (r'\\div\b',                         ' divided by '),
        (r'\\bmod\b',                        ' mod '),
        (r'\\mod\b',                         ' mod '),
        (r'\\pmod\{(.+?)\}',                 r' mod \1 '),
        (r'\\pm\b',                          ' plus or minus '),
        (r'\\infty\b',                       ' infinity '),
        (r'\\frac\{(.+?)\}\{(.+?)\}',        r'\1 over \2'),
        (r'\\dfrac\{(.+?)\}\{(.+?)\}',       r'\1 over \2'),
        (r'\^\{(.+?)\}',                     r' to the power of \1 '),
        (r'\^([0-9a-zA-Z])',                 r' to the power of \1 '),
        (r'_\{(.+?)\}',                      r' sub \1 '),
        (r'_([0-9a-zA-Z])',                  r' sub \1 '),
        (r'\\sqrt\{(.+?)\}',                 r'square root of \1'),
        (r'\\sqrt\[(.+?)\]\{(.+?)\}',        r'\1 root of \2'),
        (r'\\sum\b',                         ' sum '),
        (r'\\prod\b',                        ' product '),
        (r'\\lim\b',                         ' limit '),
        (r'\\min\b',                         ' minimum '),
        (r'\\max\b',                         ' maximum '),
        (r'\\gcd\b',                         ' gcd '),
        (r'\\lcm\b',                         ' lcm '),
        (r'\\log\b',                         ' log '),
        (r'\\ln\b',                          ' ln '),
        (r'\\in\b',                          ' in '),
        (r'\\notin\b',                       ' not in '),
        (r'\\cup\b',                         ' union '),
        (r'\\cap\b',                         ' intersection '),
        (r'\\forall\b',                      ' for all '),
        (r'\\exists\b',                      ' there exists '),
        (r'\\rightarrow\b',                  ' implies '),
        (r'\\Rightarrow\b',                  ' implies '),
        (r'\\to\b',                          ' to '),
        (r'\\binom\{(.+?)\}\{(.+?)\}',       r'\1 choose \2'),
        (r'\\dbinom\{(.+?)\}\{(.+?)\}',      r'\1 choose \2'),
        (r'\\textbf\{(.+?)\}',               r'\1'),
        (r'\\textit\{(.+?)\}',               r'\1'),
        (r'\\text\{(.+?)\}',                 r'\1'),
        (r'\\mathrm\{(.+?)\}',               r'\1'),
        (r'\\mathbf\{(.+?)\}',               r'\1'),
        (r'\\overline\{(.+?)\}',             r'\1'),
        (r'\\ldots\b',                       ' ... '),
        (r'\\cdots\b',                       ' ... '),
        (r'\\dots\b',                        ' ... '),
        (r'\\lfloor\s*(.+?)\s*\\rfloor',     r'floor of \1'),
        (r'\\lceil\s*(.+?)\s*\\rceil',       r'ceil of \1'),
        (r'\\left\b',                        ''),
        (r'\\right\b',                       ''),
        (r'\\big\w*\b',                      ''),
        (r'\\displaystyle\b',                ''),
        (r'\\([a-zA-Z]+)',                   r'\1'),
    ]

    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    text = text.replace('{', ' ').replace('}', ' ')
    text = text.replace('$', ' ')
    text = re.sub(r'\s*&\s*', ' ', text)
    text = re.sub(r'[ \t]+',  ' ', text)
    text = re.sub(r'\n{3,}',  '\n\n', text)
    return text.strip()


# ── Label utilities ────────────────────────────────────────────────────────────

def fetch_files_per_label(label, randomize=True, N=None):
    with open(LABEL_INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)
    files = index[label]["files"]
    if randomize:
        random.shuffle(files)
    if N is None:
        return files
    return files[:min(N, len(files))]


def build_multihot_label(tags: list) -> np.ndarray:
    """Return an 8-dim binary vector for the given list of tags."""
    vec = np.zeros(len(CHOSEN_LABELS), dtype=np.float32)
    for i, label in enumerate(CHOSEN_LABELS):
        if label in tags:
            vec[i] = 1.0
    return vec


# ── Dataset ────────────────────────────────────────────────────────────────────

class CodeforceDataset(Dataset):
    """One JSON file → one sample (text encoding + multi-hot label)."""

    def __init__(self, data_dir, file_names, tokenizer, domain, max_length=512):
        self.data_dir   = Path(data_dir)
        self.file_names = file_names
        self.tokenizer  = tokenizer
        self.domain     = domain
        self.max_length = max_length
        self.preprocess = preprocess_code if domain == "source_code" else preprocess_text
        self._debug_printed = False
        print(f"Dataset: {len(file_names)} files | domain: '{domain}'")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        with open(self.data_dir / self.file_names[idx], "r", encoding="utf-8") as f:
            record = json.load(f)

        text = record[self.domain]
        cleaned  = self.preprocess(text)
        
        # Print exactly once
        if not self._debug_printed:
            print("\n" + "=" * 80)
            print("RAW SAMPLE")
            print("=" * 80)
            print(text[:2000])

            print("\n" + "=" * 80)
            print("CLEANED SAMPLE")
            print("=" * 80)
            print(cleaned[:2000])

            print("\n" + "=" * 80)
            self._debug_printed = True


        n_tokens = len(self.tokenizer.tokenize(cleaned))
        if n_tokens > self.max_length - 2:   # -2 for [CLS] and [SEP]
            print(f"  WARNING: {self.file_names[idx]} has {n_tokens} tokens "
                f"— truncated to {self.max_length - 2}")
            
        encoding = self.tokenizer(
            cleaned,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        label = build_multihot_label(record["tags"])

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label),
        }


# ── Mean pooling ───────────────────────────────────────────────────────────────

def mean_pooling(token_embeddings: torch.Tensor,
                 attention_mask:   torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


# ── Embedding extraction ───────────────────────────────────────────────────────

def extract_embeddings(data_dir, file_names, label_id, tokenizer, model, device,
                       domain, max_length, batch_size=16):
    dataset = CodeforceDataset(data_dir, file_names, tokenizer, domain, max_length)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            emb            = mean_pooling(outputs.last_hidden_state, attention_mask)
            all_embeddings.append(emb.cpu().numpy())

    embeddings_np = np.vstack(all_embeddings)
    labels_np     = np.full(len(embeddings_np), label_id, dtype=np.int64)
    print(f"  label {label_id} ({CHOSEN_LABELS[label_id]}): {len(embeddings_np)} embeddings")
    return embeddings_np, labels_np


def extract_all_labels(list_of_lists, data_dir, tokenizer, model, device,
                       domain, max_length, save_dir, batch_size=16):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_embeddings, all_labels = [], []

    for label_id, file_names in enumerate(list_of_lists):
        print(f"\nLabel {label_id + 1}/{len(list_of_lists)} — {len(file_names)} files")
        emb, lbl = extract_embeddings(
            data_dir=data_dir, file_names=file_names, label_id=label_id,
            tokenizer=tokenizer, model=model, device=device,
            domain=domain, max_length=max_length, batch_size=batch_size
        )
        all_embeddings.append(emb)
        all_labels.append(lbl)

    all_embeddings = np.vstack(all_embeddings)
    all_labels     = np.concatenate(all_labels)

    np.save(save_path / "embeddings.npy", all_embeddings)
    np.save(save_path / "labels.npy",     all_labels)

    print(f"\nSaved → {save_path}  shape: {all_embeddings.shape}")
    return all_embeddings, all_labels


def plot_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    model_name,
    label_names: list = None,
    method: str = "pca",
    n_components: int = 2,
    save_path: str = None,

):
    """
    Interactive PCA / UMAP visualization with Plotly.
    """

    assert n_components in (2, 3)
    assert method in ("pca", "umap")

    print("Standardizing embeddings...")
    embeddings = StandardScaler().fit_transform(embeddings)

    print(f"Running {method.upper()} → {n_components}D ...")

    if method == "pca":
        reducer = PCA(
            n_components=n_components,
            random_state=42
        )

        projected = reducer.fit_transform(embeddings)

        explained = reducer.explained_variance_ratio_
        print(f"Explained variance: {explained.round(3)}")
        print(f"Total explained:    {explained.sum():.3f}")

    else:
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
        )

        projected = reducer.fit_transform(embeddings)

    unique_labels = sorted(np.unique(labels))

    if label_names is None:
        label_names = [f"Label {i}" for i in unique_labels]

    label_map = {
        lbl: label_names[i]
        for i, lbl in enumerate(unique_labels)
    }

    label_strings = [
        label_map[l]
        for l in labels
    ]

    # --------------------------
    # Build dataframe
    # --------------------------

    if n_components == 2:

        df = pd.DataFrame(
            {
                "x": projected[:, 0],
                "y": projected[:, 1],
                "label": label_strings,
            }
        )

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="label",
            title=f"{method.upper()} 2D — {model_name} Embeddings by Label",
            opacity=0.5,      # <-- lower opacity
            hover_data=["label"],
        )

        fig.update_traces(
            marker=dict(
                size=7,
            )
        )

    else:

        df = pd.DataFrame(
            {
                "x": projected[:, 0],
                "y": projected[:, 1],
                "z": projected[:, 2],
                "label": label_strings,
            }
        )

        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="label",
            title=f"{method.upper()} 3D — {model_name} Embeddings by Label",
            opacity=0.5,      # <-- even lower for 3D
            hover_data=["label"],
        )

        fig.update_traces(
            marker=dict(
                size=6,
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=800,
        width=1200,
        legend_title="Label",
    )

    if save_path:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)

        print(f"Plot saved → {save_path}")

    fig.show(renderer="browser")