
import os
import csv
import time

from dotenv import load_dotenv
from groq import Groq
from datasets import load_dataset

# Experiment settings
MODEL = "openai/gpt-oss-120b"
RESULTS_FILE = "structured_reasoning_results.csv"

START = 50
END = 100
MAX_REQUESTS = 10


load_dotenv()
client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
    max_retries=0
)

dataset = load_dataset(
    "super-dainiu/medagents-benchmark",
    "MedQA",
    split="test_hard"
)

columns = [
    "Question_Number", "Benchmark_ID", "Model",
    "Prediction", "Correct", "Is_Correct",
    "Response_Time", "Input_Tokens",
    "Output_Tokens", "Total_Tokens"
]

completed = set()

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            raise ValueError("Existing CSV columns do not match")
        for row in reader:
            completed.add(int(row["Question_Number"]))

requests_made = 0

with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)

    if f.tell() == 0:
        writer.writeheader()
        f.flush()

    for i in range(START, END):
        number = i + 1

        if number in completed:
            continue

        if requests_made >= MAX_REQUESTS:
            break

        sample = dataset[i]

        options = "\n".join(
            f"{key}. {value}"
            for key, value in sample["options"].items()
        )

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
            print("Stopping safely.")
            break

        elapsed = time.perf_counter() - start
        requests_made += 1

        prediction = (
            response.choices[0].message.content or ""
        ).strip().upper()

        correct = sample["answer_idx"]
        valid = prediction in sample["options"]
        is_correct = valid and prediction == correct

        usage = response.usage

        writer.writerow({
            "Question_Number": number,
            "Benchmark_ID": sample["realidx"],
            "Model": MODEL,
            "Prediction": prediction,
            "Correct": correct,
            "Is_Correct": is_correct,
            "Response_Time": round(elapsed, 3),
            "Input_Tokens": usage.prompt_tokens,
            "Output_Tokens": usage.completion_tokens,
            "Total_Tokens": usage.total_tokens
        })

        f.flush()

        print(
            f"Q{number}: {prediction} | "
            f"Correct: {correct} | Match: {is_correct}"
        )

print("New API requests:", requests_made)
print("Results saved to:", RESULTS_FILE)
