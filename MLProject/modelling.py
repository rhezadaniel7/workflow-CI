import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
mlflow.set_tracking_uri(tracking_uri)

if not os.getenv("MLFLOW_EXPERIMENT_ID"):
    mlflow.set_experiment("Latihan-Telco-Customer-Churn")

# Perbaikan Path:
# Jangan pakai path absolut (C:\Users\...). Gunakan path relatif.
# Pastikan file CSV berada di folder yang sama atau subfolder proyek.
data_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(data_path)

# Hilangkan kolom identitas yang tidak relevan untuk pelatihan model
df = df.drop(columns=['customerID'])

# Kolom 'TotalCharges' terbaca sebagai string dan memiliki beberapa nilai kosong,
# sehingga perlu dikonversi ke numerik dan nilai kosong diisi dengan median.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Encode seluruh kolom bertipe kategorikal (object) menjadi numerik
# agar dapat diproses oleh RandomForestClassifier.
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Target prediksi adalah kolom 'Churn' (bukan 'Outcome')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Mengaktifkan Autologging
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Basic"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model dilatih dengan autolog. Test accuracy: {accuracy:.4f}")
