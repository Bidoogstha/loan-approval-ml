import json

m = json.load(open("models/metrics.json"))
for name, info in m["models"].items():
    auc = info["metrics"]["roc_auc"]
    f1 = info["metrics"]["f1"]
    print(f"{name:14s} ROC-AUC={auc:.4f}  F1={f1:.4f}")