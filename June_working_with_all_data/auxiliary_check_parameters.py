import pandas as pd
import re
import os
from scipy.stats import zscore

path = "U:\\Users\\Silvia\\RutiFrishman_2025_hormones_paper\\Personality_Prediction_March_2026\\June_pareto_all_data\\"
file1 = os.path.join(path, "total_data_behaviour_hormone_2026_with_66R.xlsx")
file2 = os.path.join(path, "Data_behaviour_per_day.xlsx")

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

def normalize(name):
    """
    Convert column names to a common format.
    Example:
    'Chasing (N events)' -> 'chasingnevents'
    'Chasing.(N.events)' -> 'chasingnevents'
    """
    return re.sub(r'[^a-z0-9]+', '', str(name).lower())

# Map normalized names from file1 to original names
file1_map = {normalize(col): col for col in df1.columns}

selected_columns = []
missing_columns = []

for col2 in df2.columns:
    key = normalize(col2)

    if key in file1_map:
        selected_columns.append(file1_map[key])
    else:
        missing_columns.append(col2)

print(f"Matched columns: {len(selected_columns)}")
print(f"Missing columns: {len(missing_columns)}")

if missing_columns:
    print("\nMissing:")
    for col in missing_columns:
        print(col)

# Keep only columns corresponding to file2
filtered_df = df1[selected_columns]

# ---------------------------------------------------
# Z-score normalization
# ---------------------------------------------------

zscore_df = filtered_df.copy()
cols_to_normalize = zscore_df.columns[7:]

numeric_cols = zscore_df[cols_to_normalize].select_dtypes(include='number').columns

zscore_df[numeric_cols] = zscore_df[numeric_cols].apply(
    lambda x: zscore(x, nan_policy='omit')
)


# ---------------------------------------
# Save outputs
# ---------------------------------------

filtered_df.to_excel(
    os.path.join(path, "selected_columns_total_data_behaviour.xlsx"),
    index=False
)

zscore_df.to_excel(
    os.path.join(path, "selected_columns_total_data_behaviour_Zscore.xlsx"),
    index=False
)



print("\nSaved:")
print("selected_columns_total_data_behaviour.xlsx")