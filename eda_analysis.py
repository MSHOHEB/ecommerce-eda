"""
E-Commerce Sales Dashboard & EDA
=================================
Full Exploratory Data Analysis script.
Generates all charts to the /charts folder.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = os.path.join(os.path.dirname(__file__), "..")
DATA   = os.path.join(BASE, "data", "ecommerce_data.csv")
CHARTS = os.path.join(BASE, "charts")
os.makedirs(CHARTS, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────────────
PALETTE   = ["#4361EE","#7209B7","#F72585","#3A0CA3","#4CC9F0","#06D6A0"]
BG        = "#F8F9FA"
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
})

def savefig(name):
    path = os.path.join(CHARTS, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✅ saved → charts/{name}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ═══════════════════════════════════════════════════════════════════════════
print("\n📦 Loading data …")
df = pd.read_csv(DATA)
df["order_date"]    = pd.to_datetime(df["order_date"])
df["delivery_date"] = pd.to_datetime(df["delivery_date"])

print(f"   Shape        : {df.shape}")
print(f"   Null values  :\n{df.isnull().sum()[df.isnull().sum()>0]}")
print(f"   Duplicates   : {df.duplicated().sum()}")

# Fill nulls in rating (non-delivered orders have no rating)
df["customer_rating"].fillna(0, inplace=True)

month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

# ═══════════════════════════════════════════════════════════════════════════
# 2. OVERVIEW DASHBOARD  (2×3 grid)
# ═══════════════════════════════════════════════════════════════════════════
print("\n📊 Chart 1 – Overview Dashboard")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("E-Commerce Sales — Overview Dashboard", fontsize=18, fontweight="bold", y=1.01)

# 2a. Revenue by Category
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=True)
axes[0,0].barh(cat_rev.index, cat_rev.values/1e6, color=PALETTE[:len(cat_rev)])
axes[0,0].set_title("Revenue by Category (₹ Million)")
axes[0,0].set_xlabel("Revenue (₹ M)")
for bar, val in zip(axes[0,0].patches, cat_rev.values/1e6):
    axes[0,0].text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                   f"₹{val:.1f}M", va="center", fontsize=9)

# 2b. Monthly Revenue Trend
monthly = df.groupby(["order_year","order_month"])["revenue"].sum().reset_index()
monthly["period"] = monthly["order_year"].astype(str) + "-" + monthly["order_month"]
monthly["month_num"] = pd.to_datetime(monthly["order_month"], format="%B").dt.month
monthly = monthly.sort_values(["order_year","month_num"])
for yr, grp in monthly.groupby("order_year"):
    axes[0,1].plot(grp["order_month"], grp["revenue"]/1e6, marker="o",
                   label=str(yr), linewidth=2)
axes[0,1].set_title("Monthly Revenue Trend (₹ M)")
axes[0,1].set_xlabel("Month")
axes[0,1].set_ylabel("Revenue (₹ M)")
axes[0,1].tick_params(axis="x", rotation=45)
axes[0,1].legend()

# 2c. Order Status Pie
status_counts = df["order_status"].value_counts()
axes[0,2].pie(status_counts.values, labels=status_counts.index,
              autopct="%1.1f%%", colors=PALETTE, startangle=140,
              wedgeprops={"edgecolor":"white","linewidth":2})
axes[0,2].set_title("Order Status Distribution")

# 2d. Payment Methods
pay = df["payment_method"].value_counts()
axes[1,0].bar(pay.index, pay.values, color=PALETTE)
axes[1,0].set_title("Orders by Payment Method")
axes[1,0].set_ylabel("Number of Orders")
axes[1,0].tick_params(axis="x", rotation=30)

# 2e. Top 10 Cities by Revenue
city_rev = df.groupby("city")["revenue"].sum().nlargest(10).sort_values()
axes[1,1].barh(city_rev.index, city_rev.values/1e6, color=PALETTE[1])
axes[1,1].set_title("Top 10 Cities by Revenue (₹ M)")
axes[1,1].set_xlabel("Revenue (₹ M)")

# 2f. Quarterly Revenue
q_rev = df.groupby(["order_year","order_quarter"])["revenue"].sum().unstack("order_quarter")
q_rev.plot(kind="bar", ax=axes[1,2], color=PALETTE[:4], edgecolor="white")
axes[1,2].set_title("Quarterly Revenue by Year")
axes[1,2].set_xlabel("Year")
axes[1,2].set_ylabel("Revenue (₹)")
axes[1,2].tick_params(axis="x", rotation=0)
axes[1,2].legend(title="Quarter")

plt.tight_layout()
savefig("01_overview_dashboard.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3. PRODUCT & CATEGORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("📊 Chart 2 – Product & Category Analysis")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Product & Category Deep Dive", fontsize=16, fontweight="bold")

# 3a. Orders per Category
cat_orders = df["category"].value_counts()
axes[0].bar(cat_orders.index, cat_orders.values, color=PALETTE)
axes[0].set_title("Orders per Category")
axes[0].set_ylabel("Order Count")
axes[0].tick_params(axis="x", rotation=30)

# 3b. Avg Revenue per Order by Category
cat_avg = df.groupby("category")["revenue"].mean().sort_values(ascending=False)
axes[1].bar(cat_avg.index, cat_avg.values, color=PALETTE[2])
axes[1].set_title("Avg Order Value by Category (₹)")
axes[1].set_ylabel("Avg Revenue (₹)")
axes[1].tick_params(axis="x", rotation=30)
for bar in axes[1].patches:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                 f"₹{bar.get_height():,.0f}", ha="center", fontsize=8)

# 3c. Top 10 Products by Revenue
top_products = df.groupby("product_name")["revenue"].sum().nlargest(10).sort_values()
axes[2].barh(top_products.index, top_products.values/1e5, color=PALETTE[3])
axes[2].set_title("Top 10 Products by Revenue (₹ Lakh)")
axes[2].set_xlabel("Revenue (₹ L)")

plt.tight_layout()
savefig("02_product_category_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════
# 4. CUSTOMER BEHAVIOUR
# ═══════════════════════════════════════════════════════════════════════════
print("📊 Chart 3 – Customer Behaviour")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Customer Behaviour Analysis", fontsize=16, fontweight="bold")

# 4a. Rating Distribution (delivered only)
delivered = df[df["order_status"]=="Delivered"]
rating_counts = delivered["customer_rating"].value_counts().sort_index()
axes[0].bar(rating_counts.index.astype(int), rating_counts.values,
            color=PALETTE, edgecolor="white", width=0.6)
axes[0].set_title("Customer Rating Distribution")
axes[0].set_xlabel("Rating (1-5 ⭐)")
axes[0].set_ylabel("Number of Orders")

# 4b. Avg Rating by Category
avg_rating = delivered.groupby("category")["customer_rating"].mean().sort_values(ascending=False)
axes[1].bar(avg_rating.index, avg_rating.values, color=PALETTE[4])
axes[1].set_title("Avg Customer Rating by Category")
axes[1].set_ylabel("Rating")
axes[1].set_ylim(0, 5.5)
axes[1].tick_params(axis="x", rotation=30)
for bar in axes[1].patches:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f"{bar.get_height():.2f}", ha="center", fontsize=9)

# 4c. Repeat Customers
repeat = df.groupby("customer_id")["order_id"].count()
repeat_dist = repeat.value_counts().head(8)
axes[2].bar(repeat_dist.index.astype(str), repeat_dist.values, color=PALETTE[5])
axes[2].set_title("Repeat Customer Distribution")
axes[2].set_xlabel("Number of Orders per Customer")
axes[2].set_ylabel("Customer Count")

plt.tight_layout()
savefig("03_customer_behaviour.png")

# ═══════════════════════════════════════════════════════════════════════════
# 5. DISCOUNT & REVENUE CORRELATION
# ═══════════════════════════════════════════════════════════════════════════
print("📊 Chart 4 – Discount & Revenue")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Discount Impact on Revenue", fontsize=16, fontweight="bold")

# 5a. Revenue by Discount %
disc_rev = df.groupby("discount_pct")["revenue"].mean()
axes[0].plot(disc_rev.index, disc_rev.values, marker="o", color=PALETTE[0],
             linewidth=2.5, markersize=8)
axes[0].fill_between(disc_rev.index, disc_rev.values, alpha=0.15, color=PALETTE[0])
axes[0].set_title("Avg Revenue vs Discount %")
axes[0].set_xlabel("Discount %")
axes[0].set_ylabel("Avg Revenue (₹)")

# 5b. Discount distribution
disc_counts = df["discount_pct"].value_counts().sort_index()
axes[1].bar(disc_counts.index.astype(str), disc_counts.values, color=PALETTE[2])
axes[1].set_title("Orders by Discount Percentage")
axes[1].set_xlabel("Discount %")
axes[1].set_ylabel("Number of Orders")

plt.tight_layout()
savefig("04_discount_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════
# 6. GEOGRAPHICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("📊 Chart 5 – Geo Analysis")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Geographical Sales Analysis", fontsize=16, fontweight="bold")

# 6a. State-wise Revenue Heatmap data as bar
state_rev = df.groupby("state")["revenue"].sum().sort_values(ascending=False)
axes[0].bar(state_rev.index, state_rev.values/1e6, color=PALETTE)
axes[0].set_title("State-wise Revenue (₹ M)")
axes[0].set_ylabel("Revenue (₹ M)")
axes[0].tick_params(axis="x", rotation=45)

# 6b. City-wise Avg Delivery Days
city_del = df.groupby("city")["delivery_days"].mean().sort_values()
axes[1].barh(city_del.index, city_del.values, color=PALETTE[3])
axes[1].set_title("Avg Delivery Days by City")
axes[1].set_xlabel("Avg Delivery Days")
axes[1].axvline(city_del.mean(), color="red", linestyle="--", label=f"Mean: {city_del.mean():.1f} days")
axes[1].legend()

plt.tight_layout()
savefig("05_geo_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════
# 7. CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
print("📊 Chart 6 – Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Numerical Features — Correlation Heatmap", fontsize=16, fontweight="bold")
num_cols = ["unit_price","quantity","discount_pct","discount_amount","revenue","delivery_days","customer_rating"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, cbar_kws={"shrink":0.8})
ax.set_title("")
plt.tight_layout()
savefig("06_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# 8. KEY METRICS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n📈 KEY METRICS SUMMARY")
print("="*50)
print(f"  Total Orders      : {len(df):,}")
print(f"  Total Revenue     : ₹{df['revenue'].sum()/1e6:.2f} Million")
print(f"  Avg Order Value   : ₹{df['revenue'].mean():,.2f}")
print(f"  Unique Customers  : {df['customer_id'].nunique():,}")
print(f"  Avg Rating        : {delivered['customer_rating'].mean():.2f} ⭐")
print(f"  Delivery Rate     : {(df['order_status']=='Delivered').mean()*100:.1f}%")
print(f"  Return Rate       : {(df['order_status']=='Returned').mean()*100:.1f}%")
print(f"  Top Category      : {df.groupby('category')['revenue'].sum().idxmax()}")
print(f"  Top City          : {df.groupby('city')['revenue'].sum().idxmax()}")
print(f"  Top Payment       : {df['payment_method'].value_counts().idxmax()}")
print("="*50)
print("\n✅ All charts saved to /charts folder!")
