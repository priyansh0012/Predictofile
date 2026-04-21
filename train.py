import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Sample dataset
data = {
    "age": [18,19,20,18,21,19,22,20,18,21],
    "attendance": [85,60,90,40,75,55,92,30,80,50],
    "marks": [78,50,88,35,65,45,91,25,70,40],
    "study_hours": [3,2,4,1,2.5,1.5,5,1,3,2],
    "parent_income": [30000,20000,40000,15000,25000,18000,50000,12000,28000,20000],
    "dropout": [0,1,0,1,0,1,0,1,0,1]
}

df = pd.DataFrame(data)

X = df.drop("dropout", axis=1)
y = df["dropout"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained ✅")