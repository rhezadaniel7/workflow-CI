"""Training script untuk MLflow Project — pakai dataset preprocessed."""
import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def main(n_estimators: int, max_depth: int) -> None:

    # Load preprocessed data
    train = pd.read_csv("telco_preprocessing/train_processed.csv")
    test = pd.read_csv("telco_preprocessing/test_processed.csv")

    X_train = train.drop(columns=["Churn"])
    y_train = train["Churn"]
    X_test = test.drop(columns=["Churn"])
    y_test = test["Churn"]

    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    with mlflow.start_run(run_name="rf_ci"):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        # Manual logging params
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Manual logging metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("precision", precision_score(y_test, preds))
        mlflow.log_metric("recall", recall_score(y_test, preds))
        mlflow.log_metric("f1", f1_score(y_test, preds))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, proba))

        # Log model (artefak utama untuk Docker build)
        mlflow.sklearn.log_model(model, "model")

        print(f"[INFO] accuracy={accuracy_score(y_test, preds):.4f}")
        print(f"[INFO] f1={f1_score(y_test, preds):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=20)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth)
