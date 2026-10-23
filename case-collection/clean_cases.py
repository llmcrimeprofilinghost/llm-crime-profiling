import pandas as pd

sampled_cases = pd.read_csv("./data/sampled_cases.csv")

non_empty = sampled_cases["quality"] != "empty"
cleaned_cases = sampled_cases[non_empty]

avg_length = cleaned_cases["plain_text"].str.len().mean()
avg_words = (
  cleaned_cases["plain_text"]
  .str.split()
  .str.len()
  .mean()
)

print("Total cases:", len(sampled_cases))
print("Empty cases:", (sampled_cases["quality"] == "empty").sum())
print("Filtered cases:", len(cleaned_cases))
print("Average length:", avg_length)
print("Average words:", avg_words)

cleaned_cases.to_csv("cleaned_cases.csv", index=False)
print(f"\nSaved {len(cleaned_cases)} selected cases.")