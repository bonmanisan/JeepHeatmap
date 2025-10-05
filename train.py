# ============================================================
# 🚕 Jeepney Volume Prediction Model Trainer (FINAL FIXED)
# ============================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import numpy as np

# ============================================================
# 1️⃣ Load dataset
# ============================================================
DATA_FILE = "expandedDataset_with_JeepVolume.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("❌ Dataset not found — put expandedDataset_with_JeepVolume.csv here.")

df = pd.read_csv(DATA_FILE)
df.columns = [c.lower().strip() for c in df.columns]

print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")

# ============================================================
# 2️⃣ Select features + target
# ============================================================
expected_features = ["latitude", "longitude", "stop", "dayofweek", "hour", "season", "event"]
target_col = "jeepvolume"

missing = [c for c in expected_features + [target_col] if c not in df.columns]
if missing:
    raise ValueError(f"❌ Missing columns: {missing}")

X = df[expected_features].copy()
y = df[target_col].astype(float)

# ============================================================
# 3️⃣ Clean & detect column types
# ============================================================
for col in X.columns:
    try:
        X[col] = pd.to_numeric(X[col])
    except (ValueError, TypeError):
        X[col] = X[col].astype(str)

numeric_features = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
categorical_features = [c for c in X.columns if c not in numeric_features]

print(f"🧩 Categorical columns: {categorical_features}")
print(f"🔢 Numeric columns: {numeric_features}")

for col in categorical_features:
    X[col] = X[col].fillna("Unknown")
for col in numeric_features:
    X[col] = X[col].fillna(0)

# ============================================================
# 4️⃣ Build preprocessing + model
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ],
    sparse_threshold=0
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=12
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# ============================================================
# 5️⃣ Train-test split + training
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training model...")
pipeline.fit(X_train, y_train)

# ============================================================
# 6️⃣ Evaluate
# ============================================================
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("✅ Training complete!")
print(f"📊 R² Score: {r2:.3f}")
print(f"📉 MAE: {mae:.3f}")

# ============================================================
# 7️⃣ Save model
# ============================================================
artifact = {"model": pipeline, "feature_cols": expected_features}
joblib.dump(artifact, "jeep_pipeline.joblib")
print("💾 Saved model to jeep_pipeline.joblib")
