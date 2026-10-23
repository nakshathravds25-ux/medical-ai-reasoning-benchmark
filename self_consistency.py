
import os
import csv
import time

from dotenv import load_dotenv
from groq import Groq
from datasets import load_dataset

# EXPERIMENT SETTINGS
MODEL = "openai/gpt-oss-120b"
RESULTS_FILE = "self_consistency_samples.csv"
BASELINE_FILE = "formal_baseline_results.csv"
MAX_REQUESTS = 30
NUM_SAMPLES = 3

# CONNECT TO GROQ
load_dotenv()

client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
    max_retries=0
)

# LOAD DATASET
dataset = load_dataset(
    "super-dainiu/medagents-benchmark",
    "MedQA",
    split="test_hard"
)

# LOAD BASELINE QUESTION LIST
with open(BASELINE_FILE, newline="", encoding="utf-8") as f:
    baseline = list(csv.DictReader(f))

columns = [
    "Question_Number", "Benchmark_ID", "Sample_Number",
    "Prediction", "Correct", "Is_Correct",
    "Response_Time", "Input_Tokens",
    "Output_Tokens", "Total_Tokens"
]

# FIND ALREADY COMPLETED ATTEMPTS
completed = set()

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != columns:
            raise ValueError("Results file format mismatch")

        for row in reader:
            completed.add((
                int(row["Question_Number"]),
                int(row["Sample_Number"])
            ))

requests_made = 0
stop_run = False

# SAVE RESULTS WITHOUT OVERWRITING
with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)

    if f.tell() == 0:
        writer.writeheader()
        f.flush()

    for baseline_row in sorted(
        baseline, key=lambda r: int(r["Question_Number"])
    ):
        number = int(baseline_row["Question_Number"])
        sample = dataset[number - 1]

        # VERIFY QUESTION ID
        if str(baseline_row["Benchmark_ID"]) != str(sample["realidx"]):
            raise ValueError(f"Benchmark mismatch at Q{number}")

        # FORMAT ANSWER OPTIONS
        options = "\n".join(
            f"{k}. {v}" for k, v in sample["options"].items()
        )

        # SAME PROMPT AS STRUCTURED REASONING EXPERIMENT
        prompt = f"""Answer this medical multiple-choice question.

Carefully analyze the medical evidence.
Identify the relevant clinical concepts.
Consider each answer option and eliminate
options that are inconsistent with the evidence.
Select the single best answer.

Reply with only the final option letter,
without explanation.

Question: {sample['question']}

Options:
{options}"""

        # GENERATE THREE INDEPENDENT ANSWERS
        for attempt in range(1, NUM_SAMPLES + 1):

            # SKIP SAVED ATTEMPTS
            if (number, attempt) in completed:
                continue

            # LIMIT API REQUESTS
            if requests_made >= MAX_REQUESTS:
                break

            start = time.perf_counter()

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    reasoning_effort="low"
                )

            except Exception as error:
                print("API error:", type(error).__name__)
                print("Details:", str(error))
                print("Stopping. Saved results are preserved.")

                stop_run = True
                break

            # RECORD RESPONSE TIME
            elapsed = time.perf_counter() - start
            requests_made += 1

            # EXTRACT PREDICTION
            prediction = (
                response.choices[0].message.content or ""
            ).strip().upper()

            correct = sample["answer_idx"]

            is_correct = (
                prediction in sample["options"]
                and prediction == correct
            )

            # RECORD TOKEN USAGE
            usage = response.usage

            # SAVE RESULT
            writer.writerow({
                "Question_Number": number,
                "Benchmark_ID": sample["realidx"],
                "Sample_Number": attempt,
                "Prediction": prediction,
                "Correct": correct,
                "Is_Correct": is_correct,
                "Response_Time": round(elapsed, 3),
                "Input_Tokens": usage.prompt_tokens,
                "Output_Tokens": usage.completion_tokens,
                "Total_Tokens": usage.total_tokens
            })

            f.flush()

            print(f"Q{number} Attempt {attempt}: {prediction}")
            time.sleep(2)

        # STOP THE ENTIRE EXPERIMENT AFTER AN API ERROR
        if stop_run or requests_made >= MAX_REQUESTS:
            break

print("New API requests:", requests_made)
print("Results saved to:", RESULTS_FILE)
