import pandas as pd

file = input("What is the name of the csv? ")
cases = pd.read_csv(f"./{file}")
cases[cases["summary"].notna()][["case_id", "summary"]].to_csv("simplified.csv")
