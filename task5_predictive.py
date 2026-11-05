"""Task 5 - Healthcare Predictive Analytics: 30-day readmission prediction"""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
sns.set_theme(style="whitegrid")
df = pd.read_csv("data/healthcare_patients_clean.csv")
X = pd.get_dummies(df[["age","gender","disease","dosage_mg","length_of_stay"]], drop_first=True); y = df["readmitted"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
models = {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
          "Random Forest": RandomForestClassifier(300, max_depth=6, class_weight="balanced", random_state=42)}
txt = []; best = None
for name, m in models.items():
    m.fit(Xtr, ytr); auc = roc_auc_score(yte, m.predict_proba(Xte)[:,1])
    txt.append(f"== {name} (ROC-AUC {auc:.3f}) ==\n{classification_report(yte, m.predict(Xte))}")
    if best is None or auc > best[1]: best = (name, auc, m)
name, auc, m = best; txt.append(f"Best model: {name} (AUC {auc:.3f})")
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_estimator(m, Xte, yte, ax=ax[0], cmap="Blues"); ax[0].set_title(f"Confusion Matrix - {name}")
imp = pd.Series(getattr(m, "feature_importances_", abs(getattr(m, "coef_", [[0]*X.shape[1]])[0])), index=X.columns).sort_values()
imp.plot.barh(ax=ax[1], color="#2a9d8f"); ax[1].set_title("Feature Importance")
plt.tight_layout(); plt.savefig("outputs/task5_model_results.png", dpi=150)
open("outputs/task5_results.txt","w").write("\n".join(txt)); print("\n".join(txt))
