# Diabetes Insight Regression

High-accuracy regression pipeline that predicts disease progression one year after baseline for diabetes patients using the classic scikit-learn diabetes dataset. The project emphasises efficient feature engineering, model selection, and rich evaluation visuals suitable for a portfolio showcase.

## Highlights
- 📊 **Feature pipeline** with percentile-based scaling and polynomial interactions for key biomarkers.
- 🧠 **Model zoo** comparing Elastic Net, Random Forest, Gradient Boosting, and HistGradientBoosting regressors via cross-validation.
- 🏆 **Automatic model selection** storing the best-performing regressor with calibration plots.
- 📈 **Insightful reports**: residual vs. prediction plot, feature importance chart, and metrics JSON.
- ⚙️ **CLI-first workflow**: one command to train, evaluate, and export artifacts.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train & Evaluate
```bash
python src/train.py --seed 1337 --cv 8
```
Outputs live in `artifacts/`:
- `metrics.json`: train/validation/test MAE, RMSE, R²
- `best_model.joblib`: serialized estimator pipeline
- `residuals.png`, `feature_importance.png`: publication-ready charts

## Predict on New Data
```bash
python src/predict.py --model artifacts/best_model.joblib --input sample_patients.csv --output predictions.csv
```
Supply a CSV with the original diabetes feature columns (`age`, `sex`, `bmi`, `bp`, `s1` … `s6`).

## Project Layout
```
diabetes_insight/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data.py          # dataset loading & splitting
│   ├── features.py      # preprocessing & feature engineering pipeline
│   ├── models.py        # model candidates and evaluation helpers
│   ├── train.py         # CLI entrypoint for training/evaluation
│   └── predict.py       # CLI batch prediction helper
└── artifacts/           # generated models, metrics, plots (gitignored)
```

## Roadmap Ideas
- Hyperparameter optimisation with Optuna
- SHAP value explanations for the winning model
- FastAPI or Streamlit service for interactive use

## License
MIT License © 2025 Adityaa Bandaru
