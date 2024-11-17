
import pandas as pd


sourceDataPath = "../dataSet/sms_data.csv"
finalDataPath = "../dataSet/statistic"

# Load data
df = pd.read_csv(sourceDataPath)

# Count 'ham' instances
ham_count = df[df["Label"] == "ham"].shape[0]

# Count 'spam' instances
spam_count = df[df["Label"] == "spam"].shape[0]


with open(finalDataPath, "w", encoding="utf-8", newline="") as csvfile:
    csvfile.write(f"Number of all messages: {ham_count+spam_count}")
    csvfile.write(f"Number of ham messages: {ham_count}")
    csvfile.write(f"Number of spam messages: {spam_count}")

print(f"Data saved to {finalDataPath}")
