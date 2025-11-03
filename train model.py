# train_model.py
# Train a simple RandomForest classifier on EEG features and save the model.
# Expects CSV: feature1,feature2,...,label
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import argparse
import os

parser = argparse.ArgumentParser(description="Train EEG->command model")
parser.add_argument("--data", default="data/eeg_features.csv", help="CSV with features and label")
parser.add_argument("--out", default="models/eeg_model.joblib", help="output model path")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)

df = pd.read_csv(args.data)
if "label" not in df.columns:
    raise SystemExit("CSV must contain 'label' column")

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

joblib.dump(clf, args.out)
print(f"Saved model to {args.out}")
