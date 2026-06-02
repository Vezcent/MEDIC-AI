import urllib.request, pandas as pd

url = "https://raw.githubusercontent.com/nikkithags/Detection-and-Analysis-of-Alzheimer-s-disease/master/oasis_longitudinal.csv"
out = r"E:\Medic\MEDIC-AI\oasis\o2_using\oasis_longitudinal.csv"

urllib.request.urlretrieve(url, out)
df = pd.read_csv(out)
print(f"Rows: {len(df)}, Cols: {list(df.columns)}")
grp = df["Group"].value_counts().to_dict()
print(f"Groups: {grp}")
print(f"Unique subjects: {df['Subject ID'].nunique()}")
print(f"Unique sessions: {df['MRI ID'].nunique()}")
print(df.head(3).to_string())
