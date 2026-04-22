import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# ✅ Improved Dataset (more balanced & realistic)
data = {
    "age": [18,19,20,18,21,19,22,20,18,21,19,20,23,24,22,21,20,19],
    "attendance": [85,60,90,40,75,55,92,30,80,50,88,42,70,65,78,82,60,55],
    "marks": [78,50,88,35,65,45,91,25,70,40,82,38,68,60,75,85,55,50],
    "study_hours": [3,2,4,1,2.5,1.5,5,1,3,2,4.5,1.2,2.8,2.2,3.5,4,2,1.8],
    "parent_income": [30000,20000,40000,15000,25000,18000,50000,12000,28000,20000,45000,16000,35000,30000,42000,48000,22000,21000],
    "dropout": [0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1]
}

df = pd.DataFrame(data)

# Features & Target
X = df.drop("dropout", axis=1)
y = df["dropout"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Better model (works well on small data)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model
if not os.path.exists("model"):
    os.makedirs("model")

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Debug output
print("Model trained successfully ✅")
print("Feature Importances:", model.feature_importances_)