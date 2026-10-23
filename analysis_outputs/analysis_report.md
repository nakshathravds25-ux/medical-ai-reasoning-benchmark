# MSc Medical QA Capstone — Reproducible Analysis Summary

**Dataset:** MedicalAgentsBench / MedQA `test_hard`, Q11–Q100 (n=90).

**Data integrity:** Five files passed question coverage, duplicate, answer-key, ID, correctness-flag and token-total checks.

## Accuracy and computational cost

| Method                   |   Correct | Accuracy   | CI_Lower   | CI_Upper   |   Calls_per_Question |   Mean_Total_Tokens |   Mean_API_Latency_s |
|:-------------------------|----------:|:-----------|:-----------|:-----------|---------------------:|--------------------:|---------------------:|
| Direct baseline          |        34 | 37.8%      | 28.5%      | 48.1%      |                    1 |               438.7 |                 0.8  |
| Structured prompting     |        42 | 46.7%      | 36.7%      | 56.9%      |                    1 |               488.4 |                 0.81 |
| Critic-guided refinement |        48 | 53.3%      | 43.1%      | 63.3%      |                    2 |              1010.3 |                 1.78 |
| Self-Consistency         |        42 | 46.7%      | 36.7%      | 56.9%      |                    3 |              1497.2 |                 2.78 |
| 1 agent + judge          |        46 | 51.1%      | 41.0%      | 61.2%      |                    2 |              1020.8 |                 2.1  |
| 2 agents + judge         |        44 | 48.9%      | 38.8%      | 59.0%      |                    3 |              1543.6 |                 3.14 |
| 3 agents + judge         |        43 | 47.8%      | 37.8%      | 58.0%      |                    4 |              2029.4 |                 4.1  |

## Paired comparisons with baseline

| Comparison      |   Wrong_to_Correct |   Correct_to_Wrong |   P_Exact |   P_Holm_6 |
|:----------------|-------------------:|-------------------:|----------:|-----------:|
| Structured      |                 13 |                  5 |    0.0963 |     0.0981 |
| Critic          |                 17 |                  3 |    0.0026 |     0.0155 |
| SelfConsistency |                  9 |                  1 |    0.0215 |     0.0859 |
| Judge1          |                 17 |                  5 |    0.0169 |     0.0845 |
| Judge2          |                 14 |                  4 |    0.0309 |     0.0927 |
| Judge3          |                 13 |                  4 |    0.049  |     0.0981 |

Six paired baseline comparisons were corrected using Holm's method; do not interpret unadjusted p-values as confirmatory results.

## Agent-count contrasts (exploratory)

| Reference   | Comparison   |   Wrong_to_Correct |   Correct_to_Wrong |   P_Exact |   P_Holm_3 |
|:------------|:-------------|-------------------:|-------------------:|----------:|-----------:|
| Judge1      | Judge2       |                  2 |                  4 |    0.6875 |          1 |
| Judge2      | Judge3       |                  4 |                  5 |    1      |          1 |
| Judge1      | Judge3       |                  5 |                  8 |    0.5811 |          1 |

Roles and candidate contexts change with agent count; these differences do not isolate a pure agent-count causal effect.

## Self-Consistency

- Unanimous: 54/90
- 2-to-1 majority: 28/90
- 3-way tie: 8/90

Sample 1: **41/90**. Vote (first sample breaks three-way ties): **42/90**.

## Medical categories

Category-specific results saved as `medical_category_accuracy.csv`. Treat small subgroups as exploratory.

## Methodological boundaries

- This analysis uses stored responses, not the full 862-question benchmark; results generalize only cautiously beyond these 90 MedQA questions.

- Structured prompting requested deliberate consideration but did not necessarily reveal full Chain-of-Thought; judge/agent outputs are only answer letters.

- Token counts and response latencies are proxies for computational cost, not monetary cost or actual end-to-end wall time. Rate-limit waits and manual sleep intervals are excluded.

- Critic method cost includes its original baseline call. N-agent configurations each include n experts and only the corresponding judge; 1-, 2-, and 3-agent experiments share prior expert outputs.

- Original benchmark question text/options and prompt/script provenance are not independently verified by CSV-only analysis. Invalid option letters require verification against actual option sets.

- Paired tests use the same questions; conclusions are exploratory with a 90-question sample and pre-existing, non-preregistered comparisons.

- No result here establishes clinical safety, real-world diagnostic reliability, or causal effects of agent count.


## Research-output files

- `question_level_results.csv`, `method_summary.csv`, `paired_mcnemar_tests.csv`, `changed_questions.csv`, and `figures/*.png`.

- If supplied, `medical_category_accuracy.csv` and its heatmap.
