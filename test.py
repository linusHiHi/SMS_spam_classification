import json
import pandas as pd

config_path = "./config/config.json"
with open(config_path, "r") as f:
    config = json.load(f)
sourceDataSet = config["root"]+ config["dataset"]["dir"]+config["dataset"]["cleaned_csv"]

if config["pca"]:
    dim = config["pca_dim"]
else:
    dim = config["un_pca_dim"]

with open(sourceDataSet, "r") as f:
    df = pd.read_csv(f)

a = df["Message"]
b = df.iloc[0]
c = df["Message"][0]
print(f"df[\"Message\"]:{a}\n")
print(f"df.iloc[0]:{b}\n")
print(f"df.iloc[1]:{df.iloc[1]}\n")
print(c)
