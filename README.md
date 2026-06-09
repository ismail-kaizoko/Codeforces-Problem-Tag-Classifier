# Codeforces Problem Tag Classifier

**Author:** Ismail HAMDAOUI

**Validation Criterion:** Macro-F1

## Results

| Model | Macro F1 |
|---|---|
| TF-IDF + Logistic Regression | 0.6772 |
| TF-IDF + Random Forest (grid-searched) | 0.6763 |
| RoBERTa Encoder + MLP | 0.6503 |

## Inference

`run_inference.py` accepts either a single `.json` file or a folder of `.json` files.
When a folder is passed, a full validation report is printed.

```bash
python run_inference.py "./val_set"
python run_inference.py "./val_set/sample_127.json"
```