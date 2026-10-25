# Evaluation and figure scripts for the revised XAI-EnergyGuard manuscript

These files add the analyses that the revision reports (Sections 4.4-4.6) to your existing repository.
Put the `evaluation/` folder next to `backend/` (or point the scripts at your backend folder).

## What is here
| File | Purpose |
|---|---|
| `evaluation/loso_sensitivity.py` | Leave-one-session-out XGBoost evaluation (8 sessions with >=100 readings, 105 trees, no tuning, baseline recomputed per fold) and the labelling-threshold sweeps (overconsumption limit, current noise limit, UP rule). Prints every number quoted in Sections 4.4 and 4.5. |
| `evaluation/loso_sensitivity_output.txt` | Saved output of the run used for the manuscript. |
| `evaluation/make_figures.py` | Regenerates Figures 2, 4, 5, 6, 7, 8, 9 and 10 (file `fig11_sensitivity.png` is Figure 10 in the paper) at 300 dpi from the data and the trained model; the dashboard crop (Figure 3) is made from a screenshot given in `EG_SHOT`. |

## How to run
```bash
cd backend && pip install -r requirements.txt && pip install matplotlib && python train.py     # model + metrics.json
cd ../evaluation
python loso_sensitivity.py ../backend
EG_BACKEND=../backend EG_FIGS=figures python make_figures.py
```
`make_figures.py` also needs `data/c1_classified.csv`, `c1_counterfactuals.csv` and `c1_explanations.csv`, which `python pipeline_runner.py` writes.

## Reproducibility notes
* Re-running `train.py` with xgboost 3.4.1 (your pin is 3.2.0) reproduced `metrics.json` test metrics exactly; the early-stopping round count differed slightly (109 vs 105).
* The leave-one-session-out numbers use a fixed 105 trees; change `n_estimators` in the script if you retrain with a different round count.
* Labels are rule-derived, so all scores measure reproduction of the rules (see Section 4.2 of the manuscript).
