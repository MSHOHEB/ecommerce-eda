"""
E-Commerce Dataset Generator
Generates a realistic synthetic dataset for EDA practice.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

N = 5000  # number of orders

categories = {
    "Electronics":    ["Laptop", "Smartphone", "Headphones", "Tablet", "Smartwatch", "Camera"],
    "Clothing":       ["T-Shirt", "Jeans", "Jacket", "Dress", "Shoes", "Kurta"],
    "Home & Kitchen": ["Mixer", "Pressure Cooker", "Bed Sheet", "Curtains", "Table Lamp", "Iron"],
    "Books":          ["Fiction Novel", "Self-Help", "Engineering Book", "Children Book", "Biography", "Cookbook"],
    "Sports":         ["Cricket Bat", "Football", "Yoga Mat", "Dumbbells", "Badminton Racket", "Cycle"],
    "Beauty":         ["Face Wash", "Moisturizer", "Lipstick", "Perfume", "Hair Oil", "Sunscreen"],
}

price_ranges = {
    "Electronics":    (2000, 80000),
    "Clothing":       (300,  5000),
    "Home & Kitchen": (500,  8000),
    "Books":          (100,  1200),
    "Sports":         (400,  15000),
    "Beauty":         (150,  3000),
}

cities = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
    "Nagpur", "Surat", "Indore", "Bhopal", "Patna"
]

states = {
    "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Nagpur": "Maharashtra",
    "Delhi": "Delhi", "Jaipur": "Rajasthan",
    "Bangalore": "Karnataka",
    "Hyderabad": "Telangana",
    "Chennai": "Tamil Nadu",
    "Kolkata": "West Bengal",
    "Ahmedabad": "Gujarat", "Surat": "Gujarat",
    "Lucknow": "Uttar Pradesh",
    "Indore": "Madhya Pradesh", "Bhopal": "Madhya Pradesh",
    "Patna": "Bihar"
}

payment_methods = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery", "Wallet"]
payment_weights  = [0.20, 0.18, 0.35, 0.08, 0.14, 0.05]

order_statuses = ["Delivered", "Returned", "Cancelled", "Pending"]
status_weights  = [0.78, 0.08, 0.09, 0.05]

# Generate dates — full 2 years
dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="h")

rows = []
for i in range(1, N + 1):
    cat = np.random.choice(list(categories.keys()))
    product = np.random.choice(categories[cat])
    lo, hi = price_ranges[cat]
    price = round(np.random.uniform(lo, hi), 2)
    qty   = int(np.random.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]))
    discount_pct = np.random.choice([0, 5, 10, 15, 20, 30], p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
    discount_amt = round(price * qty * discount_pct / 100, 2)
    revenue      = round(price * qty - discount_amt, 2)

    city  = np.random.choice(cities)
    state = states[city]

    order_date    = pd.Timestamp(np.random.choice(dates))
    delivery_days = int(np.random.choice([2, 3, 4, 5, 6, 7, 10], p=[0.1, 0.2, 0.25, 0.2, 0.1, 0.1, 0.05]))
    delivery_date = order_date + pd.Timedelta(days=delivery_days)

    rating = None
    status = np.random.choice(order_statuses, p=status_weights)
    if status == "Delivered":
        rating = float(np.random.choice([1, 2, 3, 4, 5], p=[0.03, 0.05, 0.12, 0.35, 0.45]))

    rows.append({
        "order_id":        f"ORD{i:05d}",
        "customer_id":     f"CUST{np.random.randint(1000, 4000):04d}",
        "order_date":      order_date.strftime("%Y-%m-%d"),
        "order_month":     order_date.strftime("%B"),
        "order_year":      order_date.year,
        "order_quarter":   f"Q{(order_date.month - 1)//3 + 1}",
        "product_name":    product,
        "category":        cat,
        "unit_price":      price,
        "quantity":        qty,
        "discount_pct":    discount_pct,
        "discount_amount": discount_amt,
        "revenue":         revenue,
        "city":            city,
        "state":           state,
        "payment_method":  np.random.choice(payment_methods, p=payment_weights),
        "order_status":    status,
        "delivery_days":   delivery_days,
        "delivery_date":   delivery_date.strftime("%Y-%m-%d"),
        "customer_rating": rating,
    })

df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(__file__), "..", "data", "ecommerce_data.csv")
df.to_csv(out, index=False)
print(f"✅ Dataset saved → {out}")
print(df.head())
print(f"\nShape: {df.shape}")
