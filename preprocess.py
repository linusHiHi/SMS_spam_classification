import json
import pandas as pd

from preprocess.CleanUp import CleanUp
from preprocess.sbert import Sbert
from preprocess.txt2csv import txt2csv

config_path = "./config/config.json"
with open(config_path, "r") as f:
    config = json.load(f)

txtPath = config["root"] + config["dataset"]["dir"] + config["dataset"]["original"]
sourceDataPath = config["root"] + config["dataset"]["dir"] + config["dataset"]["original_csv"]
desDataPath = config["root"] + config["dataset"]["dir"] + config["dataset"]["cleaned_pickle"]

print("start preprocessing...")
try:
    with open(sourceDataPath, 'r') as f:
        df = pd.read_csv(f)
except FileNotFoundError:
    print("source csv data not found.trying txt.")
    txt2csv(txtPath, sourceDataPath)
    with open(sourceDataPath, 'r') as f:
        df = pd.read_csv(f)

print("cleanup\n")
# cleanUp
cleanup = CleanUp()
df = cleanup.cleanup(df, config["data"]["text_name"])

'''
qwqqwqwqwq
'''
print("vector\n")
#vectorize
# qwq = False if config["pca"]=="False" else True
# myModel = Sbert(config["model"],pca=qwq,pca_dim=config["pca_dim"])
# df = myModel.vectorize_df(df, config["data"]["text_name"])

df[config["data"]["tag_name"]] = df[config["data"]["tag_name"]].map(
    {'spam': config["data"]["code_name"]["spam"], 'ham': config["data"]["code_name"]["ham"]})
print("write\n")
df.to_pickle(desDataPath)
