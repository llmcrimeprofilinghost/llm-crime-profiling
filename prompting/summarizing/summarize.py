import os
import json
import time
import pandas as pd

from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()

API_TOKEN = os.getenv("ANTHROPIC_API_KEY")

client = Anthropic(api_key=API_TOKEN)

cases = pd.read_csv("./cleaned_cases.csv")

with open("summary_prompt.txt", "r", encoding="utf-8") as file:
    prompt = file.read()

CHECKPOINT_PATH = "checkpoint.csv"
MAX_ATTEMPTS = 3


schema = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "A concise factual summary of the underlying criminal "
                "incident and the primary suspect's behavior. "
                "Maximum 4 sentences."
            )
        }
    },
    "required": ["summary"],
    "additionalProperties": False
}

def process_case(case_text):
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,

        thinking={
            "type": "adaptive",
            "display": "omitted",
        },

        system=prompt,

        messages=[
            {
                "role": "user",
                "content": (
                    "<court_opinion>\n"
                    + case_text +
                    "\n</court_opinion>"
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
        (block for block in response.content if getattr(block, "type", None) == "text"),
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
            print(f"Attempt {attempt}/{MAX_ATTEMPTS} failed; retrying in {delay} seconds")
            print(f"Error is: {e}")
            time.sleep(delay)


results = []
start_index = 0

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = pd.read_csv(CHECKPOINT_PATH)

    if "summary" not in checkpoint.columns:
        raise ValueError("checkpoint.csv does not contain a summary column.")

    if len(checkpoint) > len(cases):
        raise ValueError("checkpoint.csv has more rows than cleaned_cases.csv.")

    results = checkpoint["summary"].tolist()
    start_index = len(results)
    print(f"Resuming from checkpoint: {start_index}/{len(cases)} cases completed")

for i in range(start_index, len(cases)):
    case = cases.at[i, "plain_text"]

    try:
        result = process_case_with_retry(case)
        results.append(result["summary"])

        print(f"Processed {i + 1}/{len(cases)}")

    except Exception as e:
        print(f"Error processing {i + 1}: {e}")

        results.append(None)

    checkpoint_output = pd.concat(
        [cases.iloc[:len(results)].reset_index(drop=True), pd.DataFrame({"summary": results})],
        axis=1
    )
    checkpoint_output.to_csv(CHECKPOINT_PATH, index=False)

results_df = pd.DataFrame({"summary": results})

output = pd.concat(
    [cases, results_df],
    axis=1
)

output.to_csv("summarized_cases.csv", index=False)
