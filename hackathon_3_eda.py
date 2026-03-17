# ==============================
# 1 Import Libraries
# ==============================

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report, confusion_matrix


# ==============================
# 2 Load Dataset
# ==============================

conn = sqlite3.connect("university.db")

df = pd.read_sql("SELECT * FROM students", conn)

print("Dataset Loaded")
print(df.head())


# ==============================
# 3 Basic EDA
# ==============================

print("\nDataset Info")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())


# ==============================
# 4 Visualization
# ==============================

# Placement distribution
plt.figure()
sns.countplot(x="pass", data=df)
plt.title("Placement Distribution")
plt.show()


# Attendance vs placement
plt.figure()
sns.boxplot(x="pass", y="attendance", data=df)
plt.title("Attendance vs Placement")
plt.show()


# Study hours vs placement
plt.figure()
sns.boxplot(x="pass", y="study_hours", data=df)
plt.title("Study Hours vs Placement")
plt.show()


# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()


# ==============================
# 5 Feature Selection
# ==============================

X = df[['attendance','study_hours','assignments','internal_marks']]
y = df['pass']


# ==============================
# 6 Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 7 Logistic Regression (Baseline)
# ==============================

log_model = LogisticRegression(class_weight='balanced', max_iter=1000)

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

log_recall = recall_score(y_test, log_pred)

print("\nLogistic Regression Recall:", log_recall)


# ==============================
# 8 Random Forest Model
# ==============================

rf_model = RandomForestClassifier(class_weight='balanced')

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_recall = recall_score(y_test, rf_pred)

print("\nRandom Forest Recall:", rf_recall)


# ==============================
# 9 Detailed Evaluation
# ==============================

print("\nLogistic Regression Report")
print(classification_report(y_test, log_pred))

print("\nRandom Forest Report")
print(classification_report(y_test, rf_pred))


print("\nConfusion Matrix (Random Forest)")
print(confusion_matrix(y_test, rf_pred))


# ==============================
# 10 Compare Models (Recall)
# ==============================

print("\nModel Comparison based on Recall")
print("Logistic Regression Recall:", log_recall)
print("Random Forest Recall:", rf_recall)


# ==============================
# 11 Save Best Model
# ==============================

best_model = rf_model if rf_recall > log_recall else log_model

pickle.dump(best_model, open("placement_model.pkl","wb"))

print("\nBest model saved as placement_model.pkl")
