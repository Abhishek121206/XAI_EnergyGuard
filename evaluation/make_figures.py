"""Regenerates the figures of the revised manuscript from the project data.
Usage (from this folder):  EG_BACKEND=../backend python make_figures.py
Needs matplotlib, shap and the project requirements. Figures 2, 4, 7, 8 are redrawn diagrams;
Figures 5, 6, 9, 10 are computed from the data and the trained model."""
import sys, json, textwrap, warnings
warnings.filterwarnings("ignore")
import os
BK = os.environ.get("EG_BACKEND", "../backend")   # folder with features.py, data/, models/
OUT = os.environ.get("EG_FIGS", "figures"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, BK)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555", "axes.labelcolor": "#222", "xtick.color": "#333", "ytick.color": "#333",
                     "figure.dpi": 100, "savefig.dpi": 300})
C = {"NORMAL": "#6B7280", "IE": "#0072B2", "OC": "#D55E00", "UP": "#CC79A7"}
INK, MUTED = "#1f2937", "#6b7280"
DPI = 300

def box(ax, x, y, w, h, text, fc="#ffffff", ec="#374151", fs=8, bold=False, lw=1.0, color=INK, ha="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + (w/2 if ha == "center" else 0.12), y + h/2, text, ha=ha, va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.25)
def arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, lw=lw, color=color, shrinkA=0, shrinkB=0))

# ---------------- Fig: architecture ----------------
fig, ax = plt.subplots(figsize=(6.5, 3.5)); ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
for x, w, t in [(0.1, 3.5, "SENSING LAYER"), (4.0, 5.9, "PROCESSING AND INFERENCE LAYER"), (10.3, 2.6, "PRESENTATION LAYER")]:
    ax.add_patch(FancyBboxPatch((x, 0.2), w, 6.5, boxstyle="round,pad=0,rounding_size=0.15", fc="#f3f4f6", ec="#9ca3af", lw=0.8))
    ax.text(x + w/2, 6.4, t, ha="center", fontsize=7.5, fontweight="bold", color=MUTED)
box(ax, 0.3, 4.4, 3.1, 1.6, "ESP32 node (MicroPython)\nfirmware status rule:\nIDLE_WASTE / OVERCONSUMPTION\n/ NORMAL / NO_LOAD", fs=7)
box(ax, 0.3, 2.35, 3.1, 1.7, "Sensors\nACS712-30A current (5 V load)\nHC-SR04 ultrasonic presence\nDHT11 temperature, humidity", fs=7)
box(ax, 0.3, 0.45, 3.1, 1.6, "Local alerts\nred/green LED, buzzer", fs=7)
arrow(ax, 1.85, 4.05, 1.85, 4.4)
arrow(ax, 1.85, 2.05, 1.85, 2.35, style="<|-")
box(ax, 4.2, 4.9, 2.4, 1.0, "MQTT broker\n(Mosquitto)", fs=7.5, fc="#e0f2fe")
arrow(ax, 3.4, 5.2, 4.2, 5.4)
box(ax, 7.0, 4.9, 2.7, 1.0, "data_collector.py\nCSV log + live prediction", fs=7.5)
arrow(ax, 6.6, 5.4, 7.0, 5.4)
box(ax, 4.2, 0.45, 5.5, 3.7, "", fc="#ffffff")
ax.text(6.95, 3.85, "Batch XAI pipeline (pipeline_runner.py)", ha="center", fontsize=7.5, fontweight="bold", color=INK)
steps = ["feature_engineer.py\n(9 model features)", "predict.py\nXGBoost class + Isolation Forest", "xai_module.py\nSHAP top-3 reasons", "dice_module.py\ncounterfactual power", "nlg_module.py\n3 role-adapted texts"]
for i, s in enumerate(steps):
    y = 3.15 - i * 0.62
    box(ax, 4.45, y - 0.2, 5.0, 0.52, s.replace("\n", ": "), fs=6.6, ec="#9ca3af")
    if i: arrow(ax, 6.95, y + 0.32, 6.95, y + 0.52 - 0.0, lw=0.9)
arrow(ax, 8.35, 4.9, 8.35, 4.15)
box(ax, 10.5, 4.4, 2.2, 1.6, "Flask REST API\n(:5000)", fs=7.5, fc="#e0f2fe")
box(ax, 10.5, 2.2, 2.2, 1.7, "Dashboard\n(:5500)\nOverview, Anomalies,\nExplanations, Model", fs=7)
arrow(ax, 9.7, 5.4, 10.5, 5.2); arrow(ax, 9.45, 1.6, 10.5, 3.0, lw=0.9); arrow(ax, 11.6, 4.4, 11.6, 3.9)
fig.savefig(f"{OUT}/fig2_architecture.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# ---------------- Fig: ML pipeline ----------------
fig, ax = plt.subplots(figsize=(6.5, 2.7)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.4); ax.axis("off")
box(ax, 0.1, 2.0, 1.9, 1.6, "Raw reading\npower, current,\npresence,\ntemp., humidity", fs=7)
box(ax, 2.5, 2.0, 2.2, 1.6, "Feature engineering\nrolling mean/std\n(10 readings),\npower jump, hourly\nbaseline deviation", fs=6.8)
box(ax, 5.2, 3.0, 2.5, 1.5, "XGBoost classifier\nNORMAL / IE / OC / UP", fs=7.2, fc="#dbeafe")
box(ax, 5.2, 1.0, 2.5, 1.5, "Isolation Forest\n(secondary outlier score,\nfit on NORMAL only)", fs=7, fc="#f3f4f6")
box(ax, 8.2, 3.0, 2.1, 1.5, "if not NORMAL:\nSHAP top-3\nfeatures", fs=7.2, fc="#fef3c7")
box(ax, 8.2, 1.0, 2.1, 1.5, "Counterfactual\npower search\n(DiCE / grid / rule)", fs=7, fc="#fef3c7")
box(ax, 10.8, 2.0, 2.1, 1.6, "Role-adaptive\ntext: manager,\nstudent,\nadministrator", fs=7.2, fc="#dcfce7")
arrow(ax, 2.0, 2.8, 2.5, 2.8); arrow(ax, 4.7, 3.0, 5.2, 3.6); arrow(ax, 4.7, 2.5, 5.2, 1.9)
arrow(ax, 7.7, 3.75, 8.2, 3.75); arrow(ax, 7.7, 3.4, 8.2, 1.9, lw=0.9); arrow(ax, 10.3, 3.7, 10.8, 3.2); arrow(ax, 10.3, 1.8, 10.8, 2.4)
ax.text(6.45, 0.35, "The classifier labels are derived from firmware rules plus a spike rule (see Section 4), so the pipeline reproduces and explains those rules.", ha="center", fontsize=6.3, color=MUTED, style="italic")
fig.savefig(f"{OUT}/fig4_pipeline.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# ---------------- Fig: pin map ----------------
fig, ax = plt.subplots(figsize=(6.5, 3.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")
box(ax, 5.0, 0.8, 3.0, 5.6, "ESP32\nDevKit\n(MicroPython)\n\nWi-Fi -> MQTT\n(JSON, ~1 reading/s)", fs=8, bold=True, fc="#eef2ff")
left = [("DHT11 (temperature, humidity)", "GPIO 4", 5.5), ("HC-SR04 TRIG", "GPIO 27", 4.3), ("HC-SR04 ECHO", "GPIO 26", 3.4), ("ACS712-30A OUT\n(via 2:1 divider*)", "GPIO 35 (ADC)", 1.9)]
for name, pin, y in left:
    box(ax, 0.1, y - 0.35, 3.6, 0.7, name, fs=6.8, ha="center")
    arrow(ax, 3.7, y, 5.0, y, lw=1.0, style="-"); ax.text(4.35, y + 0.12, pin, ha="center", fontsize=6.3, color=MUTED)
right = [("Green LED (present)", "GPIO 18", 5.5), ("Red LED (absent)", "GPIO 23", 4.3), ("Buzzer (idle waste)", "GPIO 19", 3.1)]
for name, pin, y in right:
    box(ax, 9.3, y - 0.35, 3.6, 0.7, name, fs=6.8)
    arrow(ax, 8.0, y, 9.3, y, lw=1.0, style="-"); ax.text(8.65, y + 0.12, pin, ha="center", fontsize=6.3, color=MUTED)
box(ax, 9.3, 0.7, 3.6, 1.5, "5 V DC test load\n(through ACS712 current path)\npower = 5 V x I (fixed voltage)", fs=6.6, fc="#fef3c7")
arrow(ax, 9.3, 1.45, 3.0, 1.45, lw=0.9, style="-"); 
ax.text(6.5, 0.25, "*Divider inferred from the x2 scaling in firmware; confirm against the wiring.", ha="center", fontsize=6, color=MUTED, style="italic")
fig.savefig(f"{OUT}/fig8_pinmap.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# ---------------- Fig: confusion matrices ----------------
M = json.load(open(f"{BK}/models/metrics.json"))
cm1 = np.array(M["test"]["xgboost"]["confusion_matrix"]["matrix"]); L = ["NORMAL", "IE", "OC", "UP"]
cm2 = np.array([[1307, 0, 0, 0], [3, 2831, 0, 0], [0, 0, 2253, 3], [0, 0, 2, 139]])   # pooled LOSO (from analysis run)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.1))
for ax, cm, t in zip(axs, [cm1, cm2], ["(a) Chronological held-out test\n(n = 979; 6-8 Oct)", "(b) Leave-one-session-out, pooled\n(n = 6,538; 8 sessions)"]):
    frac = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1.15)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8.5, color="white" if frac[i, j] > 0.6 else INK)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(L, fontsize=8); ax.set_yticklabels(L, fontsize=8)
    ax.set_xlabel("Predicted class", fontsize=8.5); ax.set_ylabel("Label (rule-derived)", fontsize=8.5); ax.set_title(t, fontsize=8.5, color=INK)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig9_confusion.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# ---------------- Fig: sensitivity ----------------
from features import build_features, make_labels
raw = pd.read_csv(f"{BK}/data/c1.csv"); d = build_features(raw)
oc_lims = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0]; pres0 = d.presence_binary == 0
def anom(oc, nz=0.02):
    ie = pres0 & (d.current_A > nz); ocm = d.power_W > oc; return float((ie | ocm).mean() * 100)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.9))
ax = axs[0]; ys = [anom(o) for o in oc_lims]
ax.plot(oc_lims, ys, "-o", color=C["OC"], lw=1.6, ms=4.5)
i = oc_lims.index(1.5); ax.plot([1.5], [ys[i]], "o", ms=9, mfc="none", mec=INK, mew=1.2)
ax.annotate("operating limit\n1.5 W: 78.9%", (1.5, ys[i]), xytext=(2.4, 84), fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))
ax.set_xlabel("Overconsumption limit (W)"); ax.set_ylabel("Readings flagged as waste (%)"); ax.set_ylim(40, 95)
ax.set_title("(a) Overconsumption limit", fontsize=8.5); ax.grid(axis="y", color="#e5e7eb", lw=0.6)
ax = axs[1]; pres = d.presence_binary == 1
cols = {0.5: "#9ecae1", 1.0: "#3182bd", 2.0: "#08306b"}
for mn, col in cols.items():
    ns = [int((pres & (d.power_jump.abs() > np.maximum(m * d.prev_rolling_std_power, mn))).sum()) for m in [2, 3, 4, 5]]
    ax.plot([2, 3, 4, 5], ns, "-o", color=col, lw=1.6, ms=4.5)
    ax.text(5.08, ns[-1], f"min {mn} W", fontsize=7, va="center", color=INK)
ax.set_xlim(1.8, 6.3); ax.set_xlabel("Spike multiplier (x previous std)"); ax.set_ylabel("Readings flagged as UP (count)")
ax.set_title("(b) Unusual-pattern rule", fontsize=8.5); ax.grid(axis="y", color="#e5e7eb", lw=0.6)
ax.plot([3], [141], "o", ms=9, mfc="none", mec=INK, mew=1.2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig11_sensitivity.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# ---------------- SHAP + counterfactual + roles ----------------
import shap
from predict import Predictor
P = Predictor()
cl = pd.read_csv(f"{BK}/data/c1_classified.csv"); ex = pd.read_csv(f"{BK}/data/c1_explanations.csv"); cf = pd.read_csv(f"{BK}/data/c1_counterfactuals.csv")
pick = {k: ex[ex.waste_type == k].iloc[-1] for k in ["IE", "OC", "UP"]}
json.dump({k: dict(v.astype(str)) for k, v in pick.items()}, open(f"{OUT}/picked_events.json", "w"), indent=1)
expl = shap.TreeExplainer(P.classifier)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.0))
NICE = {"power_W": "power (W)", "current_A": "current (A)", "presence_binary": "presence", "rolling_mean_power": "rolling mean (W)", "rolling_std_power": "rolling std (W)",
        "prev_rolling_std_power": "previous rolling std (W)", "power_jump": "power jump (W)", "power_jump_score": "jump score", "deviation_from_baseline": "baseline deviation (W)"}
for ax, k in zip(axs, ["IE", "OC"]):
    rid = pick[k]["reading_id"]; row = cl[cl.reading_id == rid].iloc[[0]]; X = P.model_input(row)
    sv = np.array(expl.shap_values(X)); ci = list(P.classes).index(k)
    v = sv[0, :, ci] if sv.ndim == 3 else sv[ci][0]
    order = np.argsort(np.abs(v))[::-1][:6][::-1]
    labels = [f"{NICE[P.features[j]]} = {float(X.iloc[0, j]):.2f}" for j in order]
    ax.barh(range(len(order)), v[order], color=[C["IE"] if x > 0 else "#9ca3af" for x in v[order]] if k == "IE" else [C["OC"] if x > 0 else "#9ca3af" for x in v[order]], height=0.62)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=7.2); ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel("SHAP value (log-odds toward predicted class)", fontsize=7.5)
    ax.set_title(f"({'a' if k=='IE' else 'b'}) {k} event, {float(row.power_W.iloc[0]):.2f} W, confidence {float(row.waste_confidence.iloc[0]):.0%}", fontsize=8, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_shap.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# counterfactual curve (replicates dice_module context_to_features)
from ml_config import LOAD_VOLTAGE, ROLLING_WINDOW, SPIKE_MIN_W, SPIKE_STD_MULTIPLIER, MODEL_FEATURES
def feats(power, ctx):
    pm = ctx["prev_mean"]; f = pd.DataFrame({"power_W": power}); f["current_A"] = power / LOAD_VOLTAGE; f["presence_binary"] = ctx["presence"]
    f["rolling_mean_power"] = pm + (power - pm) / ROLLING_WINDOW; f["rolling_std_power"] = ctx["rstd"]; f["prev_rolling_std_power"] = ctx["prstd"]
    f["power_jump"] = power - pm; f["power_jump_score"] = f["power_jump"].abs() / max(ctx["prstd"], SPIKE_MIN_W / SPIKE_STD_MULTIPLIER); f["deviation_from_baseline"] = power - ctx["base"]
    return f[MODEL_FEATURES]
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.9))
for ax, k in zip(axs, ["IE", "OC"]):
    rid = pick[k]["reading_id"]; row = cl[cl.reading_id == rid].iloc[0]; c = cf[cf.reading_id == rid].iloc[0]
    ctx = dict(prev_mean=row.power_W - row.power_jump, presence=row.presence_binary, rstd=row.rolling_std_power, prstd=row.prev_rolling_std_power, base=row.baseline_power)
    grid = np.linspace(0, max(row.power_W, 2.0) * 1.5, 300); pr = P.classifier.predict_proba(feats(grid, ctx)); pn = pr[:, list(P.classes).index("NORMAL")]
    ax.plot(grid, pn, color=C["NORMAL"], lw=1.8); ax.fill_between(grid, 0, pn, where=pn > 0.5, color=C["NORMAL"], alpha=0.15)
    ax.axhline(0.5, color="#bbb", lw=0.7, ls="--")
    ax.axvline(row.power_W, color=C[k], lw=1.4); ax.text(row.power_W, 1.02, f"observed\n{row.power_W:.2f} W", ha="center", va="bottom", fontsize=7, color=C[k])
    ax.axvline(c.counterfactual_power_W, color=INK, lw=1.2, ls=":"); ax.text(c.counterfactual_power_W, 0.55, f" counterfactual\n {c.counterfactual_power_W:.2f} W\n ({c.method})", fontsize=7, va="bottom", color=INK)
    ax.set_ylim(0, 1.25); ax.set_xlabel("Power of the current reading (W), history fixed"); ax.set_ylabel("Model probability of NORMAL")
    ax.set_title(f"({'a' if k=='IE' else 'b'}) {k} event (presence {'absent' if row.presence_binary==0 else 'present'})", fontsize=8.5, y=1.12)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_counterfactual.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# roles figure with real strings
e = pick["IE"]
fig, ax = plt.subplots(figsize=(6.5, 3.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
ax.text(6.5, 6.75, f"Same model output (idle equipment, {float(e.power_W):.2f} W) rendered for three audiences", ha="center", fontsize=8, color=INK, fontweight="bold")
for x, t, key, col in [(0.1, "Facility manager", "facility_manager", "#dbeafe"), (4.45, "Student", "student", "#dcfce7"), (8.8, "Administrator", "administrator", "#fef3c7")]:
    box(ax, x, 0.2, 4.1, 6.0, "", fc=col, ec="#9ca3af"); ax.text(x + 2.05, 5.85, t, ha="center", fontsize=8.5, fontweight="bold", color=INK)
    ax.text(x + 0.15, 5.45, textwrap.fill(str(e[key]), 38), ha="left", va="top", fontsize=6.5, color=INK, linespacing=1.3)
fig.savefig(f"{OUT}/fig7_roles.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# dashboard crop (privacy: remove browser tabs/bookmarks)
# dashboard screenshot (optional): crop browser chrome and save
SHOT = os.environ.get("EG_SHOT")
if SHOT and os.path.exists(SHOT):
    im = Image.open(SHOT).convert("RGB"); im.crop((0, 82, im.width, im.height)).save(f"{OUT}/fig3_dashboard.png", dpi=(300, 300))
print("done", [p for p in __import__("os").listdir(OUT)])


# ---- second pass: improved layouts for Figures 2, 7, 8 and 10 ----
import json, textwrap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os, sys
OUT=os.environ.get("EG_FIGS","figures"); BK=os.environ.get("EG_BACKEND","../backend"); INK="#1f2937"; MUTED="#6b7280"
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":300})
def box(ax,x,y,w,h,t,fc="#fff",ec="#374151",fs=7,bold=False,lw=1.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=0.12",fc=fc,ec=ec,lw=lw))
    if t: ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=INK,fontweight="bold" if bold else "normal",linespacing=1.25)
def arrow(ax,x1,y1,x2,y2,lw=1.1):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=8,lw=lw,color="#374151",shrinkA=0,shrinkB=0))
fig,ax=plt.subplots(figsize=(6.5,3.7)); ax.set_xlim(0,13.4); ax.set_ylim(0,7); ax.axis("off")
for x,w,t in [(0.1,3.6,"SENSING"),(4.1,6.0,"PROCESSING AND INFERENCE"),(10.5,2.8,"PRESENTATION")]:
    ax.add_patch(FancyBboxPatch((x,0.2),w,6.5,boxstyle="round,pad=0,rounding_size=0.15",fc="#f3f4f6",ec="#9ca3af",lw=0.8))
    ax.text(x+w/2,6.4,t,ha="center",fontsize=7,fontweight="bold",color=MUTED)
box(ax,0.25,4.35,3.3,1.7,"ESP32 node\n(MicroPython firmware)\nstatus rule: IDLE_WASTE,\nOVERCONSUMPTION,\nNORMAL, NO_LOAD",fs=6.3)
box(ax,0.25,2.3,3.3,1.7,"Sensors\nACS712-30A current\nHC-SR04 ultrasonic presence\nDHT11 temp./humidity",fs=6.3)
box(ax,0.25,0.45,3.3,1.5,"Local alerts\nred/green LED, buzzer",fs=6.3)
arrow(ax,1.9,4.0,1.9,4.35); arrow(ax,1.9,2.3,1.9,1.95)
box(ax,4.3,4.9,2.3,1.1,"MQTT broker\n(Mosquitto)",fs=6.8,fc="#e0f2fe")
box(ax,7.0,4.9,2.9,1.1,"data_collector.py\nCSV log + live prediction",fs=6.3)
arrow(ax,3.55,5.2,4.3,5.45); arrow(ax,6.6,5.45,7.0,5.45)
box(ax,4.3,0.45,5.6,3.75,"",fc="#fff")
ax.text(7.1,3.9,"Batch XAI pipeline (pipeline_runner.py)",ha="center",fontsize=6.8,fontweight="bold",color=INK)
steps=["feature_engineer.py: 9 model features","predict.py: XGBoost + Isolation Forest","xai_module.py: SHAP top-3 reasons","dice_module.py: counterfactual power","nlg_module.py: 3 role-adapted texts"]
for i,s in enumerate(steps):
    b=3.2-i*0.66; box(ax,4.55,b,5.1,0.46,s,fs=6.2,ec="#9ca3af")
    if i<4: arrow(ax,7.1,b,7.1,b-0.2,lw=0.9)
arrow(ax,8.45,4.9,8.45,4.2)
box(ax,10.7,4.4,2.4,1.5,"Flask REST API\n(port 5000)",fs=6.8,fc="#e0f2fe")
box(ax,10.7,1.9,2.4,2.0,"Dashboard\n(port 5500)\nOverview, Anomalies,\nExplanations, Model",fs=6.3)
arrow(ax,9.65,2.3,10.7,4.7,lw=0.9); arrow(ax,9.9,5.45,10.7,5.3); arrow(ax,11.9,4.4,11.9,3.9)
fig.savefig(f"{OUT}/fig2_architecture.png",bbox_inches="tight",pad_inches=0.05); plt.close(fig)

P=json.load(open(f"{OUT}/picked_events.json"))["IE"]
fig,ax=plt.subplots(figsize=(6.5,3.6)); ax.set_xlim(0,13); ax.set_ylim(0,7); ax.axis("off")
ax.text(6.5,6.75,f"Same event (idle equipment, {float(P['power_W']):.2f} W) rendered for three audiences",ha="center",fontsize=8,fontweight="bold",color=INK)
for x,t,k,c in [(0.1,"Facility manager","facility_manager","#dbeafe"),(4.45,"Student","student","#dcfce7"),(8.8,"Administrator","administrator","#fef3c7")]:
    box(ax,x,0.2,4.1,6.0,"",fc=c,ec="#9ca3af"); ax.text(x+2.05,5.8,t,ha="center",fontsize=8,fontweight="bold",color=INK)
    ax.text(x+0.2,5.3,textwrap.fill(str(P[k]),31),ha="left",va="top",fontsize=6,color=INK,linespacing=1.3)
fig.savefig(f"{OUT}/fig7_roles.png",bbox_inches="tight",pad_inches=0.05); plt.close(fig)

# ---- pin map v2
fig,ax=plt.subplots(figsize=(6.5,3.7)); ax.set_xlim(0,13.4); ax.set_ylim(0,7.4); ax.axis("off")
box(ax,5.3,0.9,2.8,6.0,"ESP32\nDevKit\n(MicroPython)\n\nWi-Fi, MQTT\nJSON,\n~1 reading/s",fs=7.5,bold=True,fc="#eef2ff")
L=[("DHT11\n(temp., humidity)","GPIO 4",6.1),("HC-SR04 TRIG","GPIO 27",4.9),("HC-SR04 ECHO","GPIO 26",3.9),("ACS712-30A OUT\n(2:1 divider*)","GPIO 35 (ADC)",2.5)]
for n,p,y in L:
    box(ax,0.1,y-0.4,3.5,0.8,n,fs=6.3); ax.plot([3.6,5.3],[y,y],color="#374151",lw=1.0)
    ax.text(4.45,y+0.15,p,ha="center",fontsize=6,color=MUTED)
R=[("Green LED\n(person present)","GPIO 18",6.1),("Red LED\n(person absent)","GPIO 23",4.9),("Buzzer\n(idle waste)","GPIO 19",3.7)]
for n,p,y in R:
    box(ax,9.8,y-0.4,3.5,0.8,n,fs=6.3); ax.plot([8.1,9.8],[y,y],color="#374151",lw=1.0)
    ax.text(8.95,y+0.15,p,ha="center",fontsize=6,color=MUTED)
box(ax,0.1,0.35,5.0,1.1,"5 V DC test load, wired through the\nACS712 current path; power = 5 V x I",fs=6.2,fc="#fef3c7")
ax.plot([2.0,2.0],[1.45,2.1],color="#374151",lw=1.0,ls="--")
ax.text(10.4,1.0,"*Divider inferred from the x2 scaling\nin the firmware; confirm against the wiring.",ha="center",fontsize=5.8,color=MUTED,style="italic")
fig.savefig(f"{OUT}/fig8_pinmap.png",bbox_inches="tight",pad_inches=0.05); plt.close(fig)

# ---- sensitivity v2 (legend instead of end labels)
sys.path.insert(0,BK)
import numpy as np, pandas as pd
from features import build_features
d=build_features(pd.read_csv(BK+"/data/c1.csv")); p0=d.presence_binary==0; pr=d.presence_binary==1
fig,axs=plt.subplots(1,2,figsize=(6.5,2.9)); ax=axs[0]
lims=[1.0,1.25,1.5,2.0,2.5,3.0,5.0]; ys=[float(((p0&(d.current_A>0.02))|(d.power_W>o)).mean()*100) for o in lims]
ax.plot(lims,ys,"-o",color="#D55E00",lw=1.6,ms=4.5); ax.plot([1.5],[ys[2]],"o",ms=9,mfc="none",mec=INK,mew=1.2)
ax.annotate(f"operating limit\n1.5 W: {ys[2]:.1f}%",(1.5,ys[2]),xytext=(2.4,84),fontsize=7.5,color=INK,arrowprops=dict(arrowstyle="-",color="#999",lw=0.8))
ax.set_xlabel("Overconsumption limit (W)",fontsize=8); ax.set_ylabel("Readings flagged as waste (%)",fontsize=8); ax.set_ylim(40,95)
ax.set_title("(a) Overconsumption limit",fontsize=8.5); ax.grid(axis="y",color="#e5e7eb",lw=0.6); ax.spines[["top","right"]].set_visible(False)
ax=axs[1]
for mn,col,ls in [(0.5,"#9ecae1","-"),(1.0,"#3182bd","-"),(2.0,"#08306b","-")]:
    ns=[int((pr&(d.power_jump.abs()>np.maximum(m*d.prev_rolling_std_power,mn))).sum()) for m in [2,3,4,5]]
    ax.plot([2,3,4,5],ns,ls,marker="o",color=col,lw=1.6,ms=4.5,label=f"minimum jump {mn} W")
ax.plot([3],[141],"o",ms=9,mfc="none",mec=INK,mew=1.2); ax.text(3.15,165,"operating rule\n(3x, 1 W): 141",fontsize=7,color=INK)
ax.set_xlabel("Spike multiplier (x previous std)",fontsize=8); ax.set_ylabel("Readings flagged as UP (count)",fontsize=8)
ax.set_title("(b) Unusual-pattern rule",fontsize=8.5); ax.grid(axis="y",color="#e5e7eb",lw=0.6); ax.legend(fontsize=6.5,frameon=False,loc="upper right"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig11_sensitivity.png",bbox_inches="tight",pad_inches=0.05); plt.close(fig)
