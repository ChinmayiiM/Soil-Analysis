import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os

# -----------------------------
# Step 1: Load dataset
# -----------------------------
csv_file = 'Soil Nutrients.csv'

if csv_file not in os.listdir():
    raise FileNotFoundError(f"File '{csv_file}' not found in the folder!")

df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()  # remove extra spaces

print("CSV Loaded Successfully!\n")
print("Columns in CSV:", df.columns, "\n")

# -----------------------------
# Step 2: Detect region/location column
# -----------------------------
possible_region_cols = ['Region', 'region', 'Location', 'location', 'Area', 'area']

region_col = None
for col in df.columns:
    if col in possible_region_cols:
        region_col = col
        break

if region_col is None:
    # pick first non-nutrient column
    for col in df.columns:
        if col not in ['Nitrogen', 'Phosphorus', 'Potassium']:
            region_col = col
            print(f"No standard region column found. Using '{region_col}' as region column.\n")
            break

if region_col is None:
    raise KeyError("No region/location column found in the CSV.")

print(f"Using '{region_col}' as the region/location column.\n")

# -----------------------------
# Step 3: Basic Analysis
# -----------------------------
print("Average Nutrients:\n", df[['Nitrogen','Phosphorus','Potassium']].mean(), "\n")

highest_N = df.loc[df['Nitrogen'].idxmax()]
lowest_P = df.loc[df['Phosphorus'].idxmin()]

print(f"Region with highest Nitrogen:\n{highest_N}\n")
print(f"Region with lowest Phosphorus:\n{lowest_P}\n")

# -----------------------------
# Step 4: Automatic Insights
# -----------------------------
print("\n--- INSIGHTS & RECOMMENDATIONS ---\n")
region_high_N = highest_N[region_col]
region_low_P = lowest_P[region_col]

print(f"✅ {region_high_N} has the highest Nitrogen level.")
print(f"⚠️ {region_low_P} has the lowest Phosphorus level.\n")

avg_N = df['Nitrogen'].mean()
avg_P = df['Phosphorus'].mean()
avg_K = df['Potassium'].mean()

print(f"Average Nutrients: Nitrogen={avg_N:.2f}, Phosphorus={avg_P:.2f}, Potassium={avg_K:.2f}")

if avg_N < 50:
    print("💡 Overall Nitrogen is low. Consider Nitrogen-rich fertilizers.")
if avg_P < 30:
    print("💡 Overall Phosphorus is low. Consider Phosphorus fertilizers.")
if avg_K < 40:
    print("💡 Overall Potassium is low. Consider Potassium fertilizers.")

# -----------------------------
# Step 5: Conditional Crop/Fertilizer Recommendations
# -----------------------------
def recommend_crop(row):
    if row['Nitrogen'] > 150 and row['Phosphorus'] > 100 and row['Potassium'] > 150:
        return '✅ Suitable for Wheat'
    elif row['Nitrogen'] < 100:
        return '💡 Add Nitrogen fertilizer'
    elif row['Phosphorus'] < 80:
        return '💡 Add Phosphorus fertilizer'
    else:
        return '✅ Suitable for Vegetables'

df['Recommendation'] = df.apply(recommend_crop, axis=1)
print("\nCrop/Fertilizer Recommendations by Region:\n", df[[region_col, 'Recommendation']], "\n")

# -----------------------------
# Step 6: Normalize Nutrients (0-1) for comparison
# -----------------------------
scaler = MinMaxScaler()
df[['Nitrogen_scaled','Phosphorus_scaled','Potassium_scaled']] = scaler.fit_transform(df[['Nitrogen','Phosphorus','Potassium']])
print("Scaled Nutrients (0-1):\n", df.head(), "\n")

# -----------------------------
# Step 7: Visualizations
# -----------------------------
# Histograms
for nutrient, color in zip(['Nitrogen','Phosphorus','Potassium'], ['green','orange','blue']):
    plt.figure(figsize=(6,4))
    sns.histplot(df[nutrient], kde=True, color=color)
    plt.title(f"{nutrient} Distribution")
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[['Nitrogen','Phosphorus','Potassium']].corr(), annot=True, cmap='coolwarm')
plt.title("Nutrient Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df[['Nitrogen','Phosphorus','Potassium']])
plt.show()

# Bar plot of average nutrients by region
avg_by_region = df.groupby(region_col)[['Nitrogen','Phosphorus','Potassium']].mean()
avg_by_region.plot(kind='bar', figsize=(10,6))
plt.title("Average Nutrients by Region/Location")
plt.ylabel("Amount")
plt.xlabel(region_col)
plt.xticks(rotation=45)
plt.show()

# Highlight regions above average
avg_nutrients = df[['Nitrogen','Phosphorus','Potassium']].mean()
above_avg = df[(df['Nitrogen']>avg_nutrients['Nitrogen']) | (df['Phosphorus']>avg_nutrients['Phosphorus']) | (df['Potassium']>avg_nutrients['Potassium'])]
print("Regions with Above-Average Nutrients:\n", above_avg[[region_col,'Nitrogen','Phosphorus','Potassium']], "\n")