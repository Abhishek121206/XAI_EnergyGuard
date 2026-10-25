"""Leave-one-session-out evaluation and labelling-threshold sensitivity for XAI-EnergyGuard.
Usage:  python loso_sensitivity.py <backend_dir>   (folder containing features.py, ml_config.py and data/c1.csv)
Prints every number used in Sections 4.4 and 4.5 of the revised manuscript."""
import sys, os, warnings
warnings.filterwarnings("ignore")
DATA_DIR = sys.argv[1]
sys.path.insert(0, DATA_DIR)
import numpy as np, pandas as pd
from features import build_features, make_labels, compute_baseline, apply_baseline
from ml_config import MODEL_FEATURES, CLASSES
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

_csv = os.path.join(DATA_DIR, "data", "c1.csv")
if not os.path.exists(_csv):
    _csv = os.path.join(DATA_DIR, "c1.csv")
raw = pd.read_csv(_csv)
f = build_features(raw)
f["label"] = make_labels(f)
idx = {c: i for i, c in enumerate(CLASSES)}

print("=== Label counts per session ===")
print(pd.crosstab(f["session_id"], f["label"]).to_string())

print("\n=== Leave-one-session-out (sessions >=100 rows held out; fixed 105 trees, no tuning) ===")
sizes = f.groupby("session_id").size()
rows = []; allp = []
for s in sizes.index:
    if sizes[s] < 100:
        continue
    tr = f[f.session_id != s].copy(); te = f[f.session_id == s].copy()
    b = compute_baseline(tr); apply_baseline(tr, b); apply_baseline(te, b)
    missing = [c for c in CLASSES if c not in set(tr.label)]
    if missing:
        print("skip session", s, "train missing", missing); continue
    y = tr.label.map(idx).to_numpy()
    m = XGBClassifier(objective="multi:softprob", num_class=4, n_estimators=105, max_depth=4,
                      learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                      eval_metric="mlogloss", random_state=42)
    m.fit(tr[MODEL_FEATURES].astype(float), y, sample_weight=compute_sample_weight("balanced", y))
    pred = np.array(CLASSES)[m.predict(te[MODEL_FEATURES].astype(float))]
    te["pred"] = pred
    allp.append(te[["session_id", "label", "pred"]])
    rows.append((s, len(te), round(accuracy_score(te.label, pred), 4),
                 round(f1_score(te.label, pred, labels=CLASSES, average="macro", zero_division=0), 4)))
print(pd.DataFrame(rows, columns=["held_out_session", "n", "accuracy", "macro_f1"]).to_string(index=False))
P = pd.concat(allp)
print("\nPooled out-of-session (n=%d): accuracy %.4f, macro-F1 %.4f" % (
    len(P), accuracy_score(P.label, P.pred),
    f1_score(P.label, P.pred, labels=CLASSES, average="macro", zero_division=0)))
print(classification_report(P.label, P.pred, labels=CLASSES, digits=4, zero_division=0))
print("Confusion (rows=true, cols=pred, order", CLASSES, ")")
print(confusion_matrix(P.label, P.pred, labels=CLASSES))

print("\n=== How much do labels depend on the thresholds? (rule recomputed from logged readings) ===")
d = f.copy()
pres = d.presence_binary == 1
def rule(oc, noise):
    ie = (d.presence_binary == 0) & (d.current_A > noise)
    oc_m = (d.power_W > oc)
    lab = pd.Series("NORMAL", index=d.index); lab[oc_m] = "OC"; lab[ie] = "IE"
    return lab
logged = d["status"].map({"IDLE_WASTE": "IE", "OVERCONSUMPTION": "OC", "NORMAL": "NORMAL", "NO_LOAD": "NORMAL"})
print("\nOverconsumption limit sweep (noise limit 0.02 A):")
out = []
for oc in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0]:
    l = rule(oc, 0.02)
    out.append((oc, round((l == "OC").mean() * 100, 1), round((l == "IE").mean() * 100, 1),
                round((l != "NORMAL").mean() * 100, 1), round((l == logged).mean() * 100, 1)))
print(pd.DataFrame(out, columns=["OC_limit_W", "%OC", "%IE", "%anomalous", "%agree_w_logged_labels"]).to_string(index=False))
print("\nCurrent noise-limit sweep (OC limit 1.5 W):")
out = []
for nz in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
    l = rule(1.5, nz)
    out.append((nz, round((l == "IE").mean() * 100, 1), round((l != "NORMAL").mean() * 100, 1), round((l == logged).mean() * 100, 1)))
print(pd.DataFrame(out, columns=["noise_A", "%IE", "%anomalous", "%agree_w_logged_labels"]).to_string(index=False))
near = pres & d.power_W.between(1.2, 1.8)
print("\nOccupied rows with power within +/-20%% of the 1.5 W limit: %d of %d (%.1f%%)" % (near.sum(), pres.sum(), near.sum() / pres.sum() * 100))
print("Absent rows with current between 0.02 and 0.05 A:", int(((d.presence_binary == 0) & d.current_A.between(0.0201, 0.05)).sum()))

print("\nUnusual-pattern (UP) rule sweep: count of occupied non-IE rows flagged")
out = []
for mult in [2.0, 3.0, 4.0, 5.0]:
    for mn in [0.5, 1.0, 2.0]:
        thr = np.maximum(mult * d.prev_rolling_std_power, mn)
        up = pres & (d.power_jump.abs() > thr)
        out.append((mult, mn, int(up.sum()), round(up.sum() / pres.sum() * 100, 2)))
print(pd.DataFrame(out, columns=["std_mult", "min_W", "n_UP", "%of_occupied"]).to_string(index=False))

print("\n=== Does the model beat a 3-rule decision list? (agreement with labels, in-sample) ===")
print("firmware-rule vs make_labels agreement: %.4f; disagreement is exactly the UP rows: %s" % (
    (rule(1.5, 0.02) == d.label).mean(), bool(((rule(1.5, 0.02) != d.label) == (d.label == "UP")).all())))
print("UP rows: %d (%.2f%% of all readings)" % ((d.label == "UP").sum(), (d.label == "UP").mean() * 100))
print("\nPresence-only check: P(label=IE | absent & load) and P(label in OC/NORMAL/UP | present):")
print(pd.crosstab(d.presence_binary, d.label).to_string())
print("\nPower stats by label (W):")
print(d.groupby("label").power_W.describe().round(3).to_string())
