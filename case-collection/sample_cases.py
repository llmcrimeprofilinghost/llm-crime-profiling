import time
import requests
import pandas as pd
import os

from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("COURT_LISTENER_API_KEY")
BASE_URL = "https://www.courtlistener.com/api/rest/v4/opinions"

# ----------------------------
# CONFIGURATION
# ----------------------------

OUTPUT_FILE = "sampled_cases.csv"
MAX_RETRIES = 5
RANDOM_SEED = 67
SAMPLE_SIZE = 600 + 600

file = input("Please input the name of the csv: ")
cases_DF = pd.read_csv(f"./data/{file}")

headers = {
    "Authorization": f"Token {API_TOKEN}"
}

# ----------------------------
# RANDOM SAMPLE
# ----------------------------

random_sample = cases_DF.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_SEED
)

resuming = os.path.exists(OUTPUT_FILE)

if resuming:
    existing = pd.read_csv(OUTPUT_FILE)

    needed = SAMPLE_SIZE - len(existing)

    if needed > 0:
        missing_cases = random_sample[
            ~random_sample["opinion_id"].isin(existing["opinion_id"])
        ].head(needed)

        sample_DF = pd.concat(
            [existing, missing_cases],
            ignore_index=True
        )

        print(f"Added {len(missing_cases)} new cases.")

    else:
        sample_DF = existing

else:
    sample_DF = random_sample

# ----------------------------
# GETTING PLAIN TEXT DESCRIPTIONS
# ----------------------------

for case in sample_DF.itertuples(index=True):

  if resuming and pd.notna(case.quality):
    continue
  
  id = case.opinion_id
  index = case.Index

  response = requests.get(
    f"{BASE_URL}/{id}",
    headers=headers
  )

  # Retry the API request
  for attempt in range(MAX_RETRIES):

      try:
          response = requests.get(
              f"{BASE_URL}/{id}",
              headers=headers,
              timeout=30
          )

          if response.status_code == 200:
              break

          print(
              f"Case {id}: attempt {attempt + 1}/{MAX_RETRIES} "
              f"failed with {response.status_code}"
          )

          time.sleep(10)

      except requests.RequestException as e:
          print(f"Case {id}: {e}")
          time.sleep(10)

  else:
      print(f"Giving up on case {id} after {MAX_RETRIES} attempts.")
      break

  opinion_data = response.json()
  plain_text = opinion_data["plain_text"]

  if plain_text:
    # replacing '\n' with spaces
    plain_text = plain_text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

    sample_DF.loc[index, "quality"] = "to be reviewed"
  else:
    sample_DF.loc[index, "quality"] = "empty"

  sample_DF.loc[index, "plain_text"] = plain_text
  sample_DF.to_csv(OUTPUT_FILE, index=False)
  print(f"Saved plain text for case {id}")
  time.sleep(3.4)

# ----------------------------
# SAVE FILES
# ----------------------------

sample_DF.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {SAMPLE_SIZE} selected cases.")
