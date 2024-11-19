import json
import pandas as pd

from preprocess.CleanUp import CleanUp
from preprocess.sbert import Sbert
from preprocess.txt2csv import txt2csv

config_path = "./config/config.json"
with open(config_path, "r") as f:
    config = json.load(f)

txtPath = config["root"]+ config["dataset"]["dir"]+config["dataset"]["original"]
sourceDataPath = config["root"]+ config["dataset"]["dir"]+config["dataset"]["original_csv"]
desDataPath = config["root"]+ config["dataset"]["dir"]+config["dataset"]["vectorized_csv"]
try:
    with open(sourceDataPath,'r') as f:
        df = pd.read_csv(f)
except FileNotFoundError:
    txt2csv(txtPath,sourceDataPath)
    with open(sourceDataPath, 'r') as f:
        df = pd.read_csv(f)

print("cleanup\n")
# cleanUp
cleanup = CleanUp()
df = cleanup.cleanup(df,config["data"]["text_name"])

print("vector\n")
#vectorize
myModel = Sbert(config["model"],pca=bool(config["pca"]),pca_dim=config["pca_dim"])
df = myModel.vectorize_df(df, config["data"]["text_name"])

print("write\n")
with open(config["root"]+ config["dataset"]["dir"]+config["dataset"]["vectorized_csv"],'w') as f:
    df.to_csv(f, index=False)