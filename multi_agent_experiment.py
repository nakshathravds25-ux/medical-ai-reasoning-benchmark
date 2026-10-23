
import csv
import os
import re
import time

from datasets import load_dataset
from dotenv import load_dotenv
from groq import Groq


# EXPERIMENT SETTINGS
MODEL = "openai/gpt-oss-120b"
BASELINE_FILE = "formal_baseline_results.csv"
RESULTS_FILE = "multi_agent_results.csv"

MAX_REQUESTS = 60
DELAY_SECONDS = 10

START_QUESTION = 11
END_QUESTION = 100


# AGENT ROLES
ROLES = [
    (
        "expert_1",
        "Analyze the question using relevant clinical and biomedical knowledge."
    ),
    (
        "expert_2",
        "Independently challenge the obvious answer and compare alternative options."
    ),
    (
        "expert_3",
        "Independently identify the decisive evidence and eliminate distractors."
    ),
]


# CSV COLUMNS
COLUMNS = [
    "Question_Number",
    "Benchmark_ID",
    "Stage",
    "Agent_Count",
    "Prediction",
    "Correct",
    "Is_Correct",
    "Raw_Response",
    "Response_Time",
    "Input_Tokens",
    "Output_Tokens",
    "Total_Tokens",
]


def request_answer(client, prompt):
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        reasoning_effort="low",
    )

    elapsed = time.perf_counter() - start

    raw = (
        response.choices[0].message.content or ""
    ).strip()

    predicted = (
        raw.upper()
        if re.fullmatch(r"[A-Za-z]", raw)
        else "INVALID"
    )

    return predicted, raw, elapsed, response.usage


def save_result(
    writer,
    file,
    number,
    benchmark_id,
    stage,
    agent_count,
    predicted,
    raw,
    elapsed,
    usage,
    correct,
    options,
):
    writer.writerow({
        "Question_Number": number,
        "Benchmark_ID": benchmark_id,
        "Stage": stage,
        "Agent_Count": agent_count,
        "Prediction": predicted,
        "Correct": correct,
        "Is_Correct": (
            predicted in options
            and predicted == correct
        ),
        "Raw_Response": raw,
        "Response_Time": round(elapsed, 3),
        "Input_Tokens": usage.prompt_tokens,
        "Output_Tokens": usage.completion_tokens,
        "Total_Tokens": usage.total_tokens,
    })

    file.flush()


def main():

    # CONNECT TO GROQ
    load_dotenv()

    client = Groq(
        api_key=os.environ["GROQ_API_KEY"],
        max_retries=0,
    )

    # LOAD MEDICAL BENCHMARK
    dataset = load_dataset(
        "super-dainiu/medagents-benchmark",
        "MedQA",
        split="test_hard",
    )

    # LOAD BASELINE RESULTS
    with open(
        BASELINE_FILE,
        newline="",
        encoding="utf-8",
    ) as file:
        baseline = list(csv.DictReader(file))

    # VERIFY ALL 90 QUESTIONS
    expected_questions = set(
        range(START_QUESTION, END_QUESTION + 1)
    )

    actual_questions = [
        int(row["Question_Number"])
        for row in baseline
    ]

    if (
        len(baseline) != len(expected_questions)
        or set(actual_questions) != expected_questions
    ):
        raise ValueError(
            "Expected exactly 90 baseline questions Q11-Q100"
        )

    # LOAD COMPLETED EXPERIMENT RECORDS
    saved = {}

    if os.path.exists(RESULTS_FILE):
        with open(
            RESULTS_FILE,
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames != COLUMNS:
                raise ValueError(
                    "Existing multi-agent CSV has unexpected columns"
                )

            for row in reader:
                key = (
                    int(row["Question_Number"]),
                    row["Stage"],
                )

                if key in saved:
                    raise ValueError(
                        f"Duplicate saved result: {key}"
                    )

                saved[key] = row

    calls = 0
    stop = False

    # APPEND RESULTS WITHOUT OVERWRITING
    with open(
        RESULTS_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS,
        )

        if file.tell() == 0:
            writer.writeheader()
            file.flush()

        # PROCESS QUESTIONS IN ORDER
        for baseline_row in sorted(
            baseline,
            key=lambda r: int(r["Question_Number"]),
        ):

            number = int(
                baseline_row["Question_Number"]
            )

            sample = dataset[number - 1]

            benchmark_id = str(sample["realidx"])

            # CHECK BENCHMARK ID
            if (
                str(baseline_row["Benchmark_ID"])
                != benchmark_id
            ):
                raise ValueError(
                    f"Question ID mismatch at Q{number}"
                )

            # FORMAT QUESTION AND OPTIONS
            options = "\n".join(
                f"{key}. {value}"
                for key, value in sample["options"].items()
            )

            question = (
                f"Question:\n{sample['question']}"
                f"\n\nOptions:\n{options}"
            )

            # USED ONLY FOR EVALUATION
            correct = sample["answer_idx"]

            candidates = []

            # EXPERT AND JUDGE STAGES
            for count, (stage, instruction) in enumerate(
                ROLES,
                start=1,
            ):

                key = (number, stage)

                # REUSE SAVED EXPERT ANSWERS
                if key in saved:
                    predicted = saved[key]["Prediction"]

                else:
                    if calls >= MAX_REQUESTS:
                        stop = True
                        break

                    prompt = (
                        "You are an independent medical "
                        "multiple-choice reasoning agent. "
                        + instruction
                        + " Consider the evidence and select "
                        "the single best option. "
                        "Return ONLY one option letter.\n\n"
                        + question
                    )

                    try:
                        (
                            predicted,
                            raw,
                            elapsed,
                            usage,
                        ) = request_answer(client, prompt)

                    except Exception as error:
                        print(
                            f"Stopped at Q{number} {stage}: "
                            f"{type(error).__name__}: {error}"
                        )
                        stop = True
                        break

                    save_result(
                        writer,
                        file,
                        number,
                        benchmark_id,
                        stage,
                        count,
                        predicted,
                        raw,
                        elapsed,
                        usage,
                        correct,
                        sample["options"],
                    )

                    calls += 1

                    print(
                        f"Q{number} {stage}: {predicted}"
                    )

                    time.sleep(DELAY_SECONDS)

                candidates.append(predicted)

                # JUDGE STAGE
                judge_stage = f"judge_{count}"
                key = (number, judge_stage)

                # SKIP COMPLETED JUDGE STAGES
                if key in saved:
                    continue

                if calls >= MAX_REQUESTS:
                    stop = True
                    break

                proposals = "\n".join(
                    f"Independent agent {idx}: {answer}"
                    for idx, answer in enumerate(
                        candidates,
                        start=1,
                    )
                )

                judge_prompt = (
                    "You are the final medical "
                    "question-answering judge. "
                    "Review the independently generated "
                    "candidate answers below. "
                    "Reconsider the question and options "
                    "yourself; candidates may be wrong. "
                    "Choose the single best option, "
                    "even if it differs from every candidate. "
                    "Return ONLY one option letter.\n\n"
                    + question
                    + "\n\nCandidate answers:\n"
                    + proposals
                )

                try:
                    (
                        predicted,
                        raw,
                        elapsed,
                        usage,
                    ) = request_answer(
                        client,
                        judge_prompt,
                    )

                except Exception as error:
                    print(
                        f"Stopped at Q{number} {judge_stage}: "
                        f"{type(error).__name__}: {error}"
                    )
                    stop = True
                    break

                save_result(
                    writer,
                    file,
                    number,
                    benchmark_id,
                    judge_stage,
                    count,
                    predicted,
                    raw,
                    elapsed,
                    usage,
                    correct,
                    sample["options"],
                )

                calls += 1

                print(
                    f"Q{number} {judge_stage}: "
                    f"{predicted} "
                    f"(using {count} agents)"
                )

                time.sleep(DELAY_SECONDS)

            if stop:
                break

    # FINAL PROGRESS
    completed_total = len(saved) + calls

    print("New API calls:", calls)
    print("Results saved to:", RESULTS_FILE)
    print(
        "Completed calls across all runs:",
        completed_total,
        "/ 540",
    )


if __name__ == "__main__":
    main()
