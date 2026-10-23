import os
import json
import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# =========================
# DEEPSEEK CLIENT
# =========================

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# =========================
# INPUTS
# =========================

PROMPT = "profiling_prompt.txt"
CSV_FILE = "summaries.csv"
TEXT_COLUMN = "summary"
OUTPUT_FILE = "deepseek.csv"

# Number of simultaneous API requests
MAX_WORKERS = 10

# =========================
# LOAD PROMPT
# =========================

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


# =========================
# PROCESS ONE CASE
# =========================

def process_case(i, text):
    """
    Process a single case.
    Returns:
        (index, result, error)
    """

    try:
        response = client.responses.create(
            model="deepseek-v4-flash",
            input=f"""
{prompt}

Crime summary:
{text}
""",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "suspect_profile",
                    "schema": SCHEMA
                }
            }
        )

        result = json.loads(response.output_text)

        return i, json.dumps(result), None

    except Exception as e:
        return i, None, str(e)


# =========================
# RUN DEEPSEEK
# =========================

def run_deepseek(csv_file, text_column, output_file):

    # -------------------------
    # Resume from existing file
    # -------------------------

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        print(f"Resuming from existing file: {output_file}")

        if "deepseek_profile" not in df.columns:
            df["deepseek_profile"] = None

    else:
        df = pd.read_csv(csv_file)
        df["deepseek_profile"] = None

    total = len(df)

    # -------------------------
    # Find unfinished cases
    # -------------------------

    pending = []

    for i in range(total):
        if pd.isna(df.at[i, "deepseek_profile"]):
            pending.append(i)

    print(f"\nTotal cases: {total}")
    print(f"Already completed: {total - len(pending)}")
    print(f"Remaining: {len(pending)}")
    print(f"Concurrent workers: {MAX_WORKERS}\n")

    if not pending:
        print("All cases already completed!")
        return

    # -------------------------
    # Process concurrently
    # -------------------------

    completed_since_save = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(
                process_case,
                i,
                df.at[i, text_column]
            ): i
            for i in pending
        }

        for future in as_completed(futures):

            i, result, error = future.result()

            if error:
                print(f"✗ Case {i + 1} failed: {error}")

            else:
                df.at[i, "deepseek_profile"] = result
                print(
                    f"✓ Case {i + 1}/{total} completed"
                )

            completed_since_save += 1

            # -------------------------
            # Save every 10 completed
            # -------------------------

            if completed_since_save >= 10:
                df.to_csv(output_file, index=False)
                print("  → Progress saved")
                completed_since_save = 0

    # -------------------------
    # Final save
    # -------------------------

    df.to_csv(output_file, index=False)

    print("\nFinished!")
    print(f"Results saved to: {output_file}")


# =========================
# START
# =========================

run_deepseek(
    csv_file=CSV_FILE,
    text_column=TEXT_COLUMN,
    output_file=OUTPUT_FILE
)