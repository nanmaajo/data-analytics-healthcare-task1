"""Task 2 - Healthcare Data Visualization"""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
df = pd.read_csv("data/healthcare_patients_clean.csv")
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
df["disease"].value_counts().plot.barh(ax=ax[0,0], color="#2a9d8f"); ax[0,0].set_title("Disease Distribution")
sns.histplot(df["age"], bins=20, kde=True, ax=ax[0,1], color="#e76f51"); ax[0,1].set_title("Age Distribution")
df["gender"].value_counts().plot.pie(ax=ax[1,0], autopct="%1.1f%%", colors=["#264653","#e9c46a"]); ax[1,0].set_title("Gender Ratio"); ax[1,0].set_ylabel("")
df["medication"].value_counts().head(10).plot.bar(ax=ax[1,1], color="#457b9d"); ax[1,1].set_title("Top 10 Medications Used"); ax[1,1].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig("outputs/task2_dashboard.png", dpi=150)
plt.figure(figsize=(9,5)); sns.boxplot(data=df, x="disease", y="age", palette="Set2"); plt.xticks(rotation=30); plt.title("Age by Disease")
plt.tight_layout(); plt.savefig("outputs/task2_age_by_disease.png", dpi=150); print("task2 charts saved")
