import argparse
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler



def run_lor_pipeline(df: pd.DataFrame, normalize: bool = True):
    # Split into features and label
    feature_cols = [col for col in df.columns if col not in ['set_type', 'label']]

    # Split data
    train_df = df[df['set_type'] == 'train']
    val_df = df[df['set_type'] == 'valid']
    test_df = df[df['set_type'] == 'test']

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(f"Check set_type values. Found: "
                         f"train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")

    X_train, y_train = train_df[feature_cols], train_df['label']
    X_val, y_val = val_df[feature_cols], val_df['label']
    X_test, y_test = test_df[feature_cols], test_df['label']

    # Merge train and validation for GridSearchCV
    X_grid = pd.concat([X_train, X_val])
    y_grid = pd.concat([y_train, y_val])

    # Normalize if required
    if normalize:
        scaler = StandardScaler()
        X_grid = scaler.fit_transform(X_grid)
        X_test = scaler.transform(X_test)

    test_fold = [-1] * len(X_train) + [0] * len(X_val)  # -1 = train, 0 = validation
    ps = PredefinedSplit(test_fold)

    # Define hyperparameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000, 2000, 5000]
    }

    grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=ps,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_grid, y_grid)
    best_model = grid.best_estimator_

    print(f"Best Hyperparameters: {grid.best_params_}")

    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_test_pred_proba)

    print(f"Test AUC: {auc_score:.4f}")
    return best_model, auc_score, grid.best_params_



def main(data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_parquet(data_path)
    df = df.fillna(-1)
    model, auc, best_params = run_lor_pipeline(df)

    # Save model
    model_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(model, model_path)

    # Save AUC and hyperparameters
    results_path = os.path.join(output_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write(f"AUC Score: {auc:.4f}\n")
        f.write("Best Hyperparameters:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")

    print(f"Saved model to {model_path}")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/mimiciv/p05_balance_by_label_numeric.parquet")
    parser.add_argument("--output_dir", type=str, default="results/lor/mimiciv/normalize")
    args = parser.parse_args()

    if args.data_path is None:
        raise ValueError("data_path is required")
    if args.output_dir is None:
        raise ValueError("output_dir is required")

    main(args.data_path, args.output_dir)
