import pandas as pd
import json

file = input("What csv do you want to extract the JSONs? ")
column = input("What column holds the profiling data? ")
output = input("What should the output file be named? ")

df = pd.read_csv(f"./results/{file}")

df[column] = df[column].apply(json.loads)
suspect_info = df[column].apply(lambda x: x["suspect-information"])
suspect_df = pd.json_normalize(suspect_info)
df = pd.concat([df.drop(columns=[column]), suspect_df], axis=1)

df.to_csv(output, index=False)