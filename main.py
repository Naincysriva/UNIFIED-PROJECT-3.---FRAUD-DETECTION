import pandas as pd
import pickle
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from scipy.stats import randint


def combine_pickle_files_to_csv(pickle_folder='dataset/data', output_csv='combined_fraud_data.csv'):
    all_dfs = []

    if not os.path.exists(pickle_folder):
        print(f"Folder '{pickle_folder}' does not exist.")
        return False

    for file in os.listdir(pickle_folder):
        if file.endswith('.pkl'):
            try:
                file_path = os.path.join(pickle_folder, file)
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
                    all_dfs.append(df)
            except Exception as e:
                print(f"Failed to load {file}: {e}")

    if not all_dfs:
        print("No pickle files loaded.")
        return False

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"CSV saved at {output_csv}")
    return True


def feature_engineering(data):
    data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'])
    
    data['high_amount_fraud_flag'] = (data['TX_AMOUNT'] > 220).astype(int)
    
    data = data.sort_values('TX_DATETIME')
    data['terminal_fraud_count_28d'] = data.groupby('TERMINAL_ID')['TX_FRAUD'].transform(lambda x: x.rolling(window=28, min_periods=1).sum())
    
    rolling_mean = data.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=14, min_periods=1).mean())
    data['customer_spike_flag'] = (data['TX_AMOUNT'] > 5 * rolling_mean).astype(int)
    
    data['customer_avg_30d'] = data.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    data['customer_max_30d'] = data.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=30, min_periods=1).max())
    data['customer_std_30d'] = data.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=30, min_periods=1).std().fillna(0))
    data['customer_tx_count_30d'] = data.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=30, min_periods=1).count())
    
    data['terminal_avg_30d'] = data.groupby('TERMINAL_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    data['terminal_max_30d'] = data.groupby('TERMINAL_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=30, min_periods=1).max())
    data['terminal_std_30d'] = data.groupby('TERMINAL_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=30, min_periods=1).std().fillna(0))
    data['terminal_tx_count_30d'] = data.groupby('TERMINAL_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=30, min_periods=1).count())
    
    data['customer_tx_count_1d'] = data.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=1, min_periods=1).count())
    data['terminal_tx_count_1d'] = data.groupby('TERMINAL_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=1, min_periods=1).count())
    
    data['TX_HOUR'] = data['TX_DATETIME'].dt.hour
    data['TX_DAY'] = data['TX_DATETIME'].dt.day
    data['TX_WEEKDAY'] = data['TX_DATETIME'].dt.weekday
    
    return data


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, classes, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def train_fraud_detection_model_with_random_search(csv_path='combined_fraud_data.csv'):
    data = pd.read_csv(csv_path)

    # Use only 1% of data to speed up tuning
    data = data.sample(frac=0.01, random_state=42)
    
    print("\nLoaded dataset sample:")
    print(data.head())
    
    data = feature_engineering(data)
    
    target_col = 'TX_FRAUD'
    drop_cols = ['TRANSACTION_ID', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_DATETIME', 'TX_FRAUD_SCENARIO']
    X = data.drop(columns=drop_cols + [target_col])
    y = data[target_col]
    
    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    
    param_dist = {
        'n_estimators': randint(50, 100),
        'max_depth': [None, 10],
        'min_samples_split': randint(2, 5),
        'min_samples_leaf': randint(1, 2),
        'class_weight': [None, 'balanced']
    }
    
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=3, scoring='f1', cv=2, verbose=2, random_state=42, n_jobs=1
    )
    random_search.fit(X_train_res, y_train_res)
    
    print(f"Best parameters: {random_search.best_params_}")
    
    best_rf = random_search.best_estimator_

    # Save the best model
    if not os.path.exists('output'):
        os.makedirs('output')
    joblib.dump(best_rf, 'output/fraud_detection_model.joblib')
    print("Model saved to output/fraud_detection_model.joblib")
    
    y_scores = best_rf.predict_proba(X_test)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_threshold = thresholds[f1_scores.argmax()]
    
    print(f"\nBest threshold for max F1: {best_threshold:.2f}")
    
    y_pred_best = (y_scores >= best_threshold).astype(int)
    
    print("\nConfusion Matrix with Best Threshold:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)
    print("\nClassification Report with Best Threshold:")
    print(classification_report(y_test, y_pred_best))

    # Plot confusion matrix and save
    plot_confusion_matrix(cm, classes=['Non-Fraud', 'Fraud'], save_path="output/confusion_matrix.png")

    # Plot feature importance and save
    plot_feature_importance(best_rf, X.columns, save_path="output/feature_importance.png")

    return best_rf, best_threshold


def load_model(model_path='output/fraud_detection_model.joblib'):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


if __name__ == "__main__":
    csv_file = 'combined_fraud_data.csv'
    if not os.path.exists(csv_file):
        combined = combine_pickle_files_to_csv(output_csv=csv_file)
        if not combined:
            print("❌ Failed to prepare dataset.")
            exit()
    model, best_threshold = train_fraud_detection_model_with_random_search(csv_path=csv_file)
