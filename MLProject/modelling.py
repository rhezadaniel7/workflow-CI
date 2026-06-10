import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
mlflow.set_tracking_uri(tracking_uri)

if not os.getenv("MLFLOW_EXPERIMENT_ID"):
    mlflow.set_experiment("Latihan-Telco-Customer-Churn")

# 3. Perbaikan Path:
# Jangan pakai path absolut (C:\Users\...). Gunakan path relatif.
# Pastikan file CSV berada di folder yang sama atau subfolder proyek.
data_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv' 
df = pd.read_csv(data_path)

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengaktifkan Autologging
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Basic"):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Model dilatih dengan autolog.")
