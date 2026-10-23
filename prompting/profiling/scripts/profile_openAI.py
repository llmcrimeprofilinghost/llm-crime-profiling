import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# INPUTS
# =========================

PROMPT = "profiling_prompt.txt"
CSV_FILE = "summaries.csv"
TEXT_COLUMN = "summary"
OUTPUT_FILE = "openai.csv"

with open(PROMPT, "r", encoding="utf-8") as file:
    prompt = file.read()

SCHEMA = {
    "type": "object",
    "properties": {
        "suspect-information": {
            "type": "object",
            "properties": {
                "age": {
                    "type": ["string", "null"],
                    "enum": [
                        "0-18y",
                        "19-24y",
                        "25-29y",
                        "30-39y",
                        "40-49y",
                        "50+y",
                        None
                    ]
                },
                "sex": {
                    "type": ["string", "null"],
                    "enum": [
                        "Male",
                        "Female",
                        None
                    ]
                },
                "economic-status": {
                    "type": ["string", "null"],
                    "enum": [
                        "Lower",
                        "Middle",
                        "Upper",
                        None
                    ]
                },
                "educational-level": {
                    "type": ["string", "null"],
                    "enum": [
                        "High school",
                        "Bachelor's",
                        "Above Bachelor's",
                        None
                    ]
                },
                "reasoning": {
                  "type": "string",
                  "description": (
                      "A brief justification of why you made your choices."
                  )
                }
            },
            "required": [
                "age",
                "sex",
                "economic-status",
                "educational-level",
                "reasoning"
            ],
            "additionalProperties": False
        }
    },
    "required": ["suspect-information"],
    "additionalProperties": False
}


def run_gpt(csv_file, text_column, output_file):

  # If an output file already exists, resume from it
  if os.path.exists(output_file):
      df = pd.read_csv(output_file)
      print(f"Resuming from existing file: {output_file}")
  else:
      df = pd.read_csv(csv_file)

      # Column to store GPT responses
      df["gpt_profile"] = None
      
  total = len(df)

  for i in range(total):

      # Skip cases that have already been processed
      if pd.notna(df.at[i, "gpt_profile"]):
          continue

      print(f"Processing {i + 1}/{total}")

      text = df.at[i, text_column]

      try:
          response = client.responses.create(
              model="gpt-5.6-luna",
              input=f"{prompt}\n\nCrime summary:\n{text}",
              text={
                  "format": {
                      "type": "json_schema",
                      "name": "suspect_profile",
                      "strict": True,
                      "schema": SCHEMA
                  }
              }
          )

          result = json.loads(response.output_text)

          # Store result
          df.at[i, "gpt_profile"] = json.dumps(result)

          # SAVE IMMEDIATELY
          df.to_csv(output_file, index=False)

          print(f"✓ Saved case {i + 1}/{total}")

      except Exception as e:
          print(f"✗ Case {i + 1} failed: {e}")

          # Save whatever progress we have before continuing
          df.to_csv(output_file, index=False)

          print("Progress saved. Moving to next case.")

  print("\nFinished!")
  print(f"Results saved to: {output_file}")


run_gpt(
    csv_file=CSV_FILE,
    text_column=TEXT_COLUMN,
    output_file=OUTPUT_FILE
)