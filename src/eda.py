import pandas as pd

df = pd.read_csv("../data/sleep.csv")

df = df.drop(columns=["Person ID"])

df[["Systolic", "Diastolic"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
df = df.drop(columns=["Blood Pressure"])

df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

categorical_cols = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]
df = pd.get_dummies(df, columns=categorical_cols)

print(df.info())
print(df.head())
