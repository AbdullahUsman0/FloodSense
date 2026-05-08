import pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'floodsense_training_data.csv')
for c in ['water_area_km2','water_area_change','water_area_pct_change','ds_idx']:
    if c in df.columns:
        print(c, '-> unique sample:', df[c].dropna().unique()[:5])
print('\nValue counts for ds_idx by flood_event:')
print(pd.crosstab(df['ds_idx'], df['flood_event']))
