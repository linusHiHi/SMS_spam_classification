import json
import os
import shutil
from os import error
from pathlib import Path

# Define the path to the JSON file
config_path = "./config.json"
backup_path = "./backup/config.json"
shutil.copy(config_path, backup_path)
# Step 1: Load the JSON file
with open(config_path, 'r') as file:
    config = json.load(file)

# Step 2: Modify the JSON data
config['root'] =  str(Path(os.path.abspath("main.py") ).parent.parent)# Update a value
       # Add a new section

# Step 3: Write the updated JSON back to the file
try:
    with open(config_path, 'w') as file: # type : SupportsWrite[str]
        file.write(json.dumps(config, indent=4))
except TypeError:
    os.replace(backup_path,config_path)
    raise

