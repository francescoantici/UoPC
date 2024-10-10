import pandas as pd
import json

def file_parser(fp, keyword = "#PJM", delimiter = "="):
    lines = filter(lambda l: l.startswith(keyword), open(fp).readlines())
    return list(map(lambda l: l.split(delimiter)[-1].replace('"', "").replace("'", "").strip(), lines))

def read_user_dataset(fp):
    if fp.endswith(".csv"):
        return pd.read_csv(fp)
    if fp.endswith(".parquet"):
        return pd.read_parquet(fp)
    if fp.endswith(".json"):
        return pd.DataFrame.from_dict(json.load(open(fb, "rb")))
    
    