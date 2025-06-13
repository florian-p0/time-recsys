#this code converts the beeradvocate data from json to csv format while removing a few columns which are unnecessary or cause problems with the json syntax

from pathlib import Path
import pandas as pd
import re
import io

base_path = Path("..") / "Datasets"

path = base_path / "beeradvocate.json"
new_path = base_path / "beeradvocate.csv"

with open(path, 'r', encoding='utf-8') as file:
    content = file.read()
    encoded = content.encode("ascii","ignore")
    content = "[" + encoded.decode().replace("'", '"') + "]"
    content = re.sub(", \"review/text\":[^\n]*","},\n",content)
    content = re.sub("\"beer/name\": [^}]*\"beer/beerId\":","\"beer/beerId\":",content)
    content = re.sub("\"review/appearance\": [^}]*\"review/overall\":","\"review/overall\":",content)
    content = re.sub("[^\"]}","\"}",content)
    content = re.sub(""",

"}
]""", "]", content)

frame = pd.read_json(io.StringIO(content))
frame.columns = ["item", "brewer", "abv", "style", "rating", "timestamp", "user"]
frame.to_csv(new_path,index=False)