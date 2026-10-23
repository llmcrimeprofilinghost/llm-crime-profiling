import os
import json
import time
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=API_TOKEN)

# Input file containing your existing summaries
cases = pd.read_csv("./summaries.csv")

# Prompt containing your crime-type definitions
with open("label_prompt.txt", "r", encoding="utf-8") as file:
    prompt = file.read()

CHECKPOINT_PATH = "checkpoint.csv"
OUTPUT_PATH = "labeled_cases.csv"

MAX_ATTEMPTS = 3


# Claude must return a JSON object containing only the label
schema = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": [
                "Homicide",
                "Assault",
                "Sexual Offense",
                "Kidnapping",
                "Robbery",
                "Theft",
                "Burglary",
                "Fraud",
                "Motor Vehicle Theft",
                "Arson",
                "Drug Offense",
                "Weapon Offense",
                "Driving Offense",
                "Other"
            ],
            "description": (
                "The crime type that best describes the primary criminal "
                "incident in the summary."
            )
        }
    },
    "required": ["label"],
    "additionalProperties": False
}


def process_case(case_text):
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=200,
        thinking={
            "type": "adaptive",
            "display": "omitted",
        },
        system=prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    "<crime_summary>\n"
                    + case_text +
                    "\n</crime_summary>"
                )
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": schema
            }
        }
    )

    text_block = next(
        (
            block
            for block in response.content
            if getattr(block, "type", None) == "text"
        ),
        None,
    )

    if text_block is None:
        raise ValueError("Claude returned no text block.")

    return json.loads(text_block.text)


def process_case_with_retry(case_text):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return process_case(case_text)

        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                raise

            delay = 2 ** attempt

            print(
                f"Attempt {attempt}/{MAX_ATTEMPTS} failed; "
                f"retrying in {delay} seconds"
            )
            print(f"Error is: {e}")

            time.sleep(delay)


# --------------------------------------------------
# Resume from checkpoint if one exists
# --------------------------------------------------

results = []
start_index = 0

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = pd.read_csv(CHECKPOINT_PATH)

    results = checkpoint["label"].tolist()
    start_index = len(results)

    print(
        f"Resuming from checkpoint: "
        f"{start_index}/{len(cases)} cases completed"
    )


# --------------------------------------------------
# Process each summary
# --------------------------------------------------

for i in range(start_index, len(cases)):

    case = cases.at[i, "summary"]

    try:
        result = process_case_with_retry(case)

        label = result["label"]
        results.append(label)

        print(
            f"Processed {i + 1}/{len(cases)}: {label}"
        )

    except Exception as e:
        print(f"Error processing {i + 1}: {e}")
        results.append(None)

    # Save checkpoint after every case
    checkpoint_output = pd.concat(
        [
            cases.iloc[:len(results)].reset_index(drop=True),
            pd.DataFrame({"label": results})
        ],
        axis=1
    )

    checkpoint_output.to_csv(
        CHECKPOINT_PATH,
        index=False
    )


# --------------------------------------------------
# Save final output
# --------------------------------------------------

output = pd.concat(
    [
        cases.reset_index(drop=True),
        pd.DataFrame({"label": results})
    ],
    axis=1
)

output.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nFinished! Output saved to {OUTPUT_PATH}")