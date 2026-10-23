# Evaluating Multi-Agent Reasoning Strategies for Complex Medical Question Answering

**MSc Data Science & BioAI capstone**  
**Benchmark:** MedicalAgentsBench — `MedQA`, `test_hard`  
**Formal evaluation:** 90 questions, Q11–Q100 (the first ten questions were used for pilots)  
**Underlying model:** `openai/gpt-oss-120b` accessed through Groq

## Project overview

This project compares seven inference-time reasoning configurations on the **same 90 medical multiple-choice questions**. It examines whether additional reasoning, review, or multiple agents improve answer accuracy and whether the improvement justifies the extra API calls, tokens, and latency.

The experiments are **complete**. You can reproduce the statistical analysis and figures using the existing CSV files **without a Groq API key or any new model calls**.

### Evaluated approaches

| Configuration | What was actually implemented | LLM calls / question |
|---|---|---:|
| Direct baseline | One model answer, no extra review | 1 |
| Structured prompting | Prompt asks model to consider evidence and alternatives; only the final answer is recorded | 1 |
| Critic-guided refinement | A separate call reviews the saved baseline answer and returns a revised answer | 2 total, including baseline |
| Self-Consistency | Three independently sampled final answers and majority voting | 3 |
| 1 specialist + judge | One independent candidate answer, then a judge | 2 |
| 2 specialists + judge | Two independent candidate answers, then a judge | 3 |
| 3 specialists + judge | Three independent candidate answers, then a judge | 4 |

All configurations use the same underlying model. Structured prompting is **not** a recorded full chain-of-thought trace; the multi-agent approach uses **candidate answers plus a judge**, not multi-round debate. The specialist candidates are reused across the agent-count configurations.

## Findings from the saved 90-question results

| Approach | Correct / 90 | Accuracy |
|---|---:|---:|
| Direct baseline | 34 | 37.8% |
| Structured prompting | 42 | 46.7% |
| Critic-guided refinement | 48 | **53.3%** |
| Self-Consistency | 42 | 46.7% |
| 1 specialist + judge | 46 | 51.1% |
| 2 specialists + judge | 44 | 48.9% |
| 3 specialists + judge | 43 | 47.8% |

Critic-guided refinement has the highest observed accuracy. Its improvement over baseline survives Holm correction of six paired comparisons (adjusted **p ≈ 0.0155**). The agent-count differences are not statistically significant; more agents did not increase observed accuracy within this implementation. See `analysis_outputs/analysis_report.md` for the details and qualifications.

## Repository layout

```text
Medical_AI_Capstone/
├── Medical_QA_Final_Analysis.ipynb     # Analysis and figure generation (run this)
├── formal_baseline_results.csv         # 90 formal baseline responses
├── structured_reasoning_results.csv    # 90 structured prompt responses
├── critic_results.csv                  # 90 review/refinement responses
├── self_consistency_samples.csv        # 270 samples: 3 per question
├── multi_agent_results.csv             # 540 stages: 6 per question
├── medical_questions_90.csv            # Q11–Q100 texts, options and identifiers
├── question_categories.csv             # Reviewed, provisional AI-assisted labels
├── question_categories_original.csv    # Original labeling snapshot
├── medical_category_codebook.md        # Classification definitions
├── category_label_review.csv           # Category review record
├── category_review_guide.md            # Ambiguities and review notes
├── medical_qa_qualitative_review.md    # Selected cases and visual limitations
├── analysis_outputs/
│   ├── analysis_report.md
│   ├── method_summary.csv
│   ├── paired_mcnemar_tests.csv
│   ├── question_level_results.csv
│   ├── changed_questions.csv
│   ├── medical_category_accuracy.csv
│   └── figures/                      # Six PNG research figures
├── baseline_research.py                # Formal baseline collection script
├── structured_reasoning.py
├── critic_refinement.py
├── self_consistency.py
├── multi_agent_experiment.py
├── auto_run_capstone.py                # Optional API-based checkpoint runner
├── export_medical_questions_90.py     # Question export from Hugging Face
├── baseline.py, baseline_batch.py      # Early pilots
└── requirements.txt                   # Python dependencies
```

Other exploratory scripts may also remain in the folder. Some experiment scripts contain the **final continuation ranges**, not the original starting ranges; do not interpret their current `START` setting as the entire experiment's scope. The complete Q11–Q100 results are stored in the CSVs.

## Reproduce the analysis (no paid service or API key required)

These instructions assume **Windows PowerShell** and Python 3.13 (the development environment used for this project).

1. Open the `Medical_AI_Capstone` directory in VS Code.
2. Activate the project's existing virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If there is no `.venv`, create one and activate it:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Open `Medical_QA_Final_Analysis.ipynb` in VS Code. Select the `.venv` Jupyter kernel (install Microsoft's **Python** and **Jupyter** extensions if requested), then choose **Run All**.
5. Read `analysis_outputs/analysis_report.md` and inspect `analysis_outputs/figures/`.

The notebook reads the five experiment CSVs, checks their coverage/keys, computes method statistics and paired comparisons, and writes results to `analysis_outputs/`. It also generates the category heatmap when `question_categories.csv` is available. **It does not send requests to Groq or re-generate model answers.**

### Verify experiment checkpoint counts without API requests

```powershell
python auto_run_capstone.py --check
```

This displays completeness counts for critic, Self-Consistency, and multi-agent records, using the two other experiment CSVs as references. **Do not run** `python auto_run_capstone.py` **without `--check`** unless you specifically intend to execute unfinished model experiments: that mode can make Groq API calls.

## Re-running model experiments (optional; not necessary for this capstone)

The original collection scripts use the Hugging Face `super-dainiu/medagents-benchmark` dataset (`MedQA`, `test_hard`) and Groq's `openai/gpt-oss-120b` model. A Groq key would be supplied locally in a `.env` file:

```text
GROQ_API_KEY=your_key_here
```

**Never commit, upload, or share `.env` or an API key.** The `.env` and `.venv` paths should remain in `.gitignore`. The study's existing responses are already saved, so re-running is unnecessary and may incur usage or rate limits. Do not overwrite the existing CSVs.

## Statistical and methodological notes

- Accuracy is computed on the same **90** questions for every configuration.
- Confidence intervals are **95% Wilson intervals**; paired accuracy comparisons use **exact McNemar tests**, with Holm adjustments for families of comparisons.
- Self-Consistency uses a first-sample tie-break when all three answers differ; this rule matters for interpretation.
- Token totals and accumulated API response durations estimate computational cost; they **exclude** explicit sleeps, rate-limit waits, and end-to-end operational overhead.
- Category labels are **AI-assisted and provisional**, not clinician-validated ground truth. Category subgroup results have small sample sizes and are exploratory.
- Some question stems refer to unavailable medical images. **Q55 is particularly ambiguous:** an option-letter/text mismatch can make an `E` response invalid against the benchmark answer key `D`. Saved scoring is preserved rather than retrospectively changed.
- The 90-question MedQA sample is **not** the full MedicalAgentsBench benchmark. Results do not demonstrate clinical safety, diagnostic validity, or a general causal effect of adding agents.
- The scripts and stored answer letters do not allow auditing complete internal model reasoning. Document exact prompt and run settings before claiming a fully independent replication.

## Main outputs for the final MSc report

Use the saved `analysis_outputs` CSV tables and six PNG figures (overall accuracy, error correction/regression, agent count versus cost, accuracy versus tokens, Self-Consistency agreement, and medical category heatmap). The report's discussion should distinguish statistically supported conclusions from exploratory patterns.

## Source and attribution

- Benchmark: [MedicalAgentsBench on Hugging Face](https://huggingface.co/datasets/super-dainiu/medagents-benchmark).
- Evaluation was carried out solely as a research comparison of multiple-choice predictions; **not** as a clinical decision-support system.

*Documentation reflects the uploaded project snapshot; it is not a claim that every experiment can be rerun identically from the saved continuation scripts alone.*
