"""Task 4 - Adverse Drug Reaction (ADR) Data Analysis"""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
df = pd.read_csv("data/adr_records.csv")
rate = (df.groupby("drug")["adr_occurred"].agg(["mean","sum","count"]).rename(columns={"mean":"adr_rate","sum":"adr_cases","count":"patients"}).sort_values("adr_rate", ascending=False))
high_risk = rate[rate["adr_rate"] > rate["adr_rate"].mean() + 0.5*rate["adr_rate"].std()]
adr = df[df["adr_occurred"]==1]
ct = pd.crosstab(adr["drug"], adr["symptom"])
df["age_group"] = pd.cut(df["age"], [0,30,45,60,100], labels=["<30","30-45","45-60","60+"])
df["dose_group"] = df["dose_mg"].map(lambda x: "High (>=250mg)" if x>=250 else "Low (<250mg)")
by_age = df.groupby("age_group", observed=True)["adr_occurred"].mean()*100
by_dose = df.groupby("dose_group")["adr_occurred"].mean()*100
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
(rate["adr_rate"]*100).plot.bar(ax=ax[0,0], color="#e76f51"); ax[0,0].set_title("ADR Rate by Drug (%)")
sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=ax[0,1]); ax[0,1].set_title("Drug vs Symptom (ADR cases)")
by_age.plot.bar(ax=ax[1,0], color="#457b9d"); ax[1,0].set_title("ADR Rate by Age Group (%)"); ax[1,0].tick_params(axis="x", rotation=0)
by_dose.plot.bar(ax=ax[1,1], color="#2a9d8f"); ax[1,1].set_title("ADR Rate by Dose (%)"); ax[1,1].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig("outputs/task4_adr_dashboard.png", dpi=150)
txt = (f"ADR rate by drug:\n{(rate['adr_rate']*100).round(1).to_string()}\n\nHigh-risk drugs: {', '.join(high_risk.index)}\n\n"
       f"Symptom-drug correlation:\n{ct.to_string()}\n\nMedian days to reaction: {adr['days_to_reaction'].median():.0f}\n\n"
       f"ADR % by age group:\n{by_age.round(1).to_string()}\n\nADR % by dose:\n{by_dose.round(1).to_string()}\n\n"
       "Possible causes: higher ADR rates in patients 60+ and at higher doses, plus drug-specific profiles (NSAIDs -> GI/skin, fluoroquinolones -> tendon/CNS, antiplatelets -> bleeding).")
open("outputs/task4_insights.txt","w").write(txt); print(txt)
