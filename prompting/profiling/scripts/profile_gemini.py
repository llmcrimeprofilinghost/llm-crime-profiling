import os
import json
import pandas as pd

from google import genai
from dotenv import load_dotenv

load_dotenv()

# =========================
# GEMINI CLIENT
# =========================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# =========================
# INPUTS
# =========================

PROMPT = "profiling_prompt.txt"
CSV_FILE = "test.csv"
TEXT_COLUMN = "summary"
OUTPUT_FILE = "gemini.csv"

with open(PROMPT, "r", encoding="utf-8") as file:
    prompt = file.read()

# =========================
# JSON SCHEMA
# =========================
SCHEMA = {
    "type": "object",
    "properties": {
        "suspect-information": {
            "type": "object",
            "properties": {

                "age": {
                    "type": "string",
                    "enum": [
                        "0-18y",
                        "19-24y",
                        "25-29y",
                        "30-39y",
                        "40-49y",
                        "50+y"
                    ]
                },

                "sex": {
                    "type": "string",
                    "enum": [
                        "Male",
                        "Female"
                    ]
                },

                "economic-status": {
                    "type": "string",
                    "enum": [
                        "Lower",
                        "Middle",
                        "Upper"
                    ]
                },

                "educational-level": {
                    "type": "string",
                    "enum": [
                        "High school",
                        "Bachelor's",
                        "Above Bachelor's"
                    ]
                },

                "reasoning": {
                    "type": "string",
                    "description": "A brief justification of why you made your choices."
                }
            },

            "required": [
                "age",
                "sex",
                "economic-status",
                "educational-level",
                "reasoning"
            ]
        }
    },

    "required": [
        "suspect-information"
    ]
}

# =========================
# RUN GEMINI
# =========================

def run_gemini(csv_file, text_column, output_file):

    # If an output file already exists, resume from it
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"Resuming from existing file: {output_file}")

    else:
        df = pd.read_csv(csv_file)

        # Column to store Gemini responses
        df["gemini_profile"] = None

    total = len(df)

    for i in range(total):

        # Skip cases that have already been processed
        if pd.notna(df.at[i, "gemini_profile"]):
            continue

        print(f"Processing {i + 1}/{total}")

        text = df.at[i, text_column]

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",

                contents=f"""
                        {prompt}

                        Crime summary:
                        {text}
                        """,

                config={
                    "response_mime_type": "application/json",
                    "response_schema": SCHEMA
                }
            )

            # Parse Gemini's JSON response
            result = json.loads(response.text)

            # Store result
            df.at[i, "gemini_profile"] = json.dumps(result)

            # SAVE IMMEDIATELY
            df.to_csv(output_file, index=False)

            print(f"✓ Saved case {i + 1}/{total}")

        except Exception as e:

            print(f"✗ Case {i + 1} failed: {e}")

            # Save whatever progress we have
            df.to_csv(output_file, index=False)

            print("Progress saved. Moving to next case.")

    print("\nFinished!")
    print(f"Results saved to: {output_file}")


# =========================
# START
# =========================

run_gemini(
    csv_file=CSV_FILE,
    text_column=TEXT_COLUMN,
    output_file=OUTPUT_FILE
)