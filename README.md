# UNIFIED-PROJECT-3.---FRAUD-DETECTION


# Fraud Transaction Detection Project

## Objective

Build a machine learning system to classify transactions as fraudulent or legitimate based on transaction details and derived features.

## Dataset

- The dataset contains simulated transactions with fraud labels according to three scenarios:
  1. Transactions over amount 220 are marked as fraud.
  2. Terminals chosen daily have all transactions marked fraud for the next 28 days.
  3. Random customers have a portion of their transactions inflated and marked fraud over 14 days.

- Data includes fields like TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT, TX_FRAUD.

## Approach

- Combined multiple raw data files into a single CSV for ease of use.
- Created engineered features based on fraud scenarios such as fraud flags based on amount, terminal fraud counts, and customer spending spikes.
- Added additional aggregated and velocity features for terminals and customers.
- Handled class imbalance using SMOTE oversampling.
- Trained and tuned a Random Forest classifier using hyperparameter search and threshold tuning.
- Saved the best model for future inference.
- Included visualizations such as confusion matrix and feature importance for explainability.

## Usage

1. Place all required `.pkl` raw data files in the `dataset/data/` folder.
2. Run the `main.py` script:
   ```
   python main.py
   ```
3. The combined dataset CSV will be created automatically if not present.
4. The model will be trained and saved in `output/fraud_detection_model.joblib`.
5. Confusion matrix and feature importance plots will be shown and saved in `output/` folder.

## Requirements

Install dependencies with:
```
pip install -r requirements.txt
```

## Future work

- Train and evaluate on larger samples or full dataset.
- Experiment with other models (XGBoost, LightGBM).
- Add model explainability techniques such as SHAP.
- Deploy model as REST API for real-time prediction.
- Analyze feature impact on specific fraud scenarios.

