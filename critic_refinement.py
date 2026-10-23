
import os
import csv
import time

from dotenv import load_dotenv
from groq import Groq
from datasets import load_dataset

# EXPERIMENT SETTINGS
MODEL = "openai/gpt-oss-120b"
BASELINE_FILE = "formal_baseline_results.csv"
RESULTS_FILE = "critic_results.csv"
MAX_REQUESTS = 10

load_dotenv()
client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
    max_retries=0
)

# Load the same benchmark questions
dataset = load_dataset(
    "super-dainiu/medagents-benchmark",
    "MedQA",
    split="test_hard"
)

# Read previously saved baseline predictions
with open(BASELINE_FILE, newline="", encoding="utf-8") as f:
    baseline = list(csv.DictReader(f))

# Find questions already reviewed
completed = set()

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            completed.add(int(row["Question_Number"]))

columns = [
    "Question_Number", "Baseline_Prediction",
    "Critic_Prediction", "Correct",
    "Baseline_Correct", "Critic_Correct",
    "Response_Time", "Input_Tokens",
    "Output_Tokens", "Total_Tokens"
]

requests_made = 0

with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)

    if f.tell() == 0:
        writer.writeheader()
        f.flush()

    for row in sorted(baseline, key=lambda r: int(r["Question_Number"])):

        number = int(row["Question_Number"])

        if number in completed:
            continue

        if requests_made >= MAX_REQUESTS:
            break

        sample = dataset[number - 1]
        original = row["Prediction"].strip().upper()
        correct = sample["answer_idx"]

        if str(row["Benchmark_ID"]) != str(sample["realidx"]):
            raise ValueError(f"Benchmark ID mismatch at Q{number}")

        options = "\n".join(
            f"{key}. {value}"
            for key, value in sample["options"].items()
        )

        prompt = f"""Review an earlier answer to this medical
multiple-choice question.

Question: {sample['question']}

Options:
{options}

Earlier answer: {original}

Independently reconsider the medical evidence and options.
Look for mistakes in the earlier answer.
Keep it if correct; otherwise select a better answer.

Reply with only the final option letter."""

        start = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                reasoning_effort="low"
            )
        except Exception as error:
            print("API error:", type(error).__name__)
            print("Stopping. Existing results are saved.")
            break

        elapsed = time.perf_counter() - start
        requests_made += 1

        revised = (
            response.choices[0].message.content or ""
        ).strip().upper()

        baseline_correct = original == correct
        critic_correct = revised == correct

        usage = response.usage

        writer.writerow({
            "Question_Number": number,
            "Baseline_Prediction": original,
            "Critic_Prediction": revised,
            "Correct": correct,
            "Baseline_Correct": baseline_correct,
            "Critic_Correct": critic_correct,
            "Response_Time": round(elapsed, 3),
            "Input_Tokens": usage.prompt_tokens,
            "Output_Tokens": usage.completion_tokens,
            "Total_Tokens": usage.total_tokens
        })

        f.flush()

        print(
            f"Q{number}: Original={original}, "
            f"Revised={revised}, Correct={correct}, "
            f"Match={critic_correct}"
        )

print("New API requests:", requests_made)
print("Results saved to:", RESULTS_FILE)
