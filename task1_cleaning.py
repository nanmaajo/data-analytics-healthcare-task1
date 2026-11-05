"""Task 1 - Healthcare Data Cleaning & Understanding"""
import pandas as pd
df = pd.read_csv("data/healthcare_patients_raw.csv", parse_dates=["admission_date"])
out = []
out.append(f"Raw shape: {df.shape}\n\nColumn types:\n{df.dtypes}\n")
out.append("Column roles: patient_id=identifier | age, dosage_mg, length_of_stay=numerical | gender, disease, medication=categorical | admission_date=date | readmitted=binary target\n")
out.append(f"Missing values:\n{df.isna().sum()}\n\nDuplicate rows: {df.duplicated().sum()}\n")

df = df.drop_duplicates()
df["gender"] = df["gender"].str.strip().str.lower().map({"male":"Male","m":"Male","female":"Female","f":"Female"})
df["age"] = df["age"].fillna(df["age"].median())
df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
df["medication"] = df["medication"].fillna("Not Recorded")
df.to_csv("data/healthcare_patients_clean.csv", index=False)

out.append(f"Clean shape: {df.shape} | remaining nulls: {int(df.isna().sum().sum())}\n")
out.append(f"Total patients: {df['patient_id'].nunique()}")
out.append(f"Average age: {df['age'].mean():.1f}")
out.append(f"Gender split:\n{df['gender'].value_counts().to_string()}")
out.append(f"Most common diseases:\n{df['disease'].value_counts().to_string()}")
out.append(f"Top medications:\n{df['medication'].value_counts().head(5).to_string()}")
out.append(f"Average age by disease:\n{df.groupby('disease')['age'].mean().round(1).to_string()}")
open("outputs/task1_summary.txt","w").write("\n".join(out)); print("\n".join(out))
