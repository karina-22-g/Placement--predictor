import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# LOAD DATA
# -----------------------------
conn = sqlite3.connect("university.db")
df = pd.read_sql("SELECT * FROM students", conn)

# -----------------------------
# BASIC CHECK
# -----------------------------
print("First 5 rows:")
print(df.head())

print("\nShape of dataset:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------
# 1️⃣ PLACEMENT DISTRIBUTION
# -----------------------------
plt.figure()
sns.countplot(x="placed", data=df)
plt.title("Placement Distribution")
plt.show()

# -----------------------------
# 2️⃣ CGPA DISTRIBUTION (IMPORTANT)
# -----------------------------
plt.figure()
sns.histplot(data=df, x="cgpa", hue="placed", bins=20, kde=True)
plt.title("CGPA vs Placement")
plt.show()

# -----------------------------
# 3️⃣ FEATURE IMPORTANCE (MAIN GRAPH)
# -----------------------------
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=["placed","student_id"])
y = df["placed"]

rf = RandomForestClassifier()
rf.fit(X, y)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance (What matters most)")
plt.show()

# -----------------------------
# 4️⃣ HEATMAP
# -----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
