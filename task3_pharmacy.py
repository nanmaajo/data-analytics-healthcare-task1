"""Task 3 - Pharmacy Sales & Drug Analysis"""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
df = pd.read_csv("data/pharmacy_sales.csv")
top = df.groupby("medicine")[["units_sold","revenue"]].sum().sort_values("revenue", ascending=False)
monthly = df.groupby("month")["revenue"].sum().to_frame(); monthly["mom_growth_%"] = monthly["revenue"].pct_change()*100
bg = df.groupby("type")[["units_sold","revenue"]].sum(); bg["revenue_share_%"] = bg["revenue"]/bg["revenue"].sum()*100
top.to_csv("outputs/task3_top_medicines.csv"); monthly.to_csv("outputs/task3_monthly_growth.csv"); bg.to_csv("outputs/task3_branded_vs_generic.csv")
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
top["revenue"].head(10).sort_values().plot.barh(ax=ax[0], color="#2a9d8f"); ax[0].set_title("Top 10 Medicines by Revenue")
monthly["revenue"].plot(ax=ax[1], marker="o", color="#e76f51"); ax[1].set_title("Monthly Revenue Trend"); ax[1].tick_params(axis="x", rotation=60)
bg["revenue"].plot.bar(ax=ax[2], color=["#264653","#e9c46a"]); ax[2].set_title("Branded vs Generic Revenue"); ax[2].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig("outputs/task3_pharmacy_dashboard.png", dpi=150)
txt = f"Top seller (revenue): {top.index[0]}\nTop seller (units): {top['units_sold'].idxmax()}\nAvg MoM growth: {monthly['mom_growth_%'].mean():.2f}%\nBest month: {monthly['revenue'].idxmax()}\n\n{bg.round(1).to_string()}"
open("outputs/task3_insights.txt","w").write(txt); print(txt)
