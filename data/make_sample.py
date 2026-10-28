"""Generates data/sample_sales.csv: a deliberately messy sales export.

Messiness on purpose (the agent must clean it): duplicated rows, missing values,
prices stored as text with '$', inconsistent region casing and spaces, mixed date formats.
"""

import random
from datetime import date, timedelta

import pandas as pd

random.seed(7)
PRODUCTS = {"Basic Plan": 29.0, "Pro Plan": 79.0, "Team Plan": 199.0, "Add-on Storage": 9.5}
REGIONS = ["LATAM", "North America", "Europe"]
rows = []
start = date(2026, 1, 1)
for i in range(600):
    d = start + timedelta(days=random.randint(0, 270))
    product = random.choices(list(PRODUCTS), weights=[5, 4, 2, 3])[0]
    region = random.choice(REGIONS)
    qty = random.randint(1, 5)
    price = PRODUCTS[product]
    rows.append(
        {
            "order_id": f"A{1000 + i}",
            "order_date": d.isoformat() if i % 7 else d.strftime("%d/%m/%Y"),
            "region": random.choice([region, region.lower(), f" {region} "]),
            "product": product,
            "quantity": qty if i % 23 else None,
            "unit_price": f"${price:.2f}" if i % 3 else price,
            "channel": random.choice(["web", "sales", "partner", None]),
        }
    )
df = pd.DataFrame(rows)
df = pd.concat([df, df.sample(18, random_state=1)])  # duplicates
df.to_csv("data/sample_sales.csv", index=False)
print(df.shape)
