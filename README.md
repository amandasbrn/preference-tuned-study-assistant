# Preference-Tuned Study Assistant using DPO

## Overview

This project builds an end-to-end preference-tuned study assistant using **Direct Preference Optimization (DPO)**.

The motivation for this project comes from my own learning experience as a data science student. When studying technical subjects such as Deep Learning, NLP, Cloud Computing, and Cybersecurity, I often use language models to help me understand difficult concepts. However, the answers are not always in the style I actually need for revision. Sometimes they are too verbose, too abstract, too formal, or not structured in a way that helps me prepare for exams.

For studying, I usually need explanations that are:

* Simple enough to understand quickly
* Technically correct
* Structured for revision
* Concise but complete
* Written in exam-ready language
* Supported by small examples or analogies when useful

This project explores whether preference tuning can help a small open-source language model better align with that preferred study style.

The final system compares a base instruction-tuned model with a DPO-tuned model trained on preference pairs of study explanations.

## Problem Statement

General-purpose language models can answer study questions, but their responses are not always aligned with how students actually study.

For example, when preparing for exams, a useful answer is not just “correct.” It also needs to be clear, memorable, structured, and easy to turn into an exam response. A technically correct but overly formal answer may still be hard to study from. On the other hand, a friendly answer that is vague or inaccurate is also not useful.

This project focuses on the following question:

> Can Direct Preference Optimization help a small language model produce study explanations that are more aligned with a student’s preferred revision style?

The target behavior is a study assistant that explains technical concepts in a way that is clear, exam-oriented, and beginner-friendly while still preserving technical accuracy.

## Why This Project Matters to Me

This project is connected to how I personally learn technical material.

During my master’s study, I often need to quickly understand concepts across machine learning, deep learning, NLP, cloud computing, and cybersecurity. I do not only want a model that gives long textbook-style explanations. I want a model that can explain concepts in the way I would want to revise them before an exam:

1. Start with a simple definition.
2. Explain the intuition.
3. Give a small example if useful.
4. End with a concise exam takeaway.

That preferred answer structure became the foundation for the preference dataset used in this project.

Rather than only fine-tuning a model to imitate answers, this project uses preference pairs to teach the model which answer style is more useful for study revision.

## Why DPO?

Direct Preference Optimization is a preference-tuning method that trains a model using pairs of responses:

* `chosen`: the preferred answer
* `rejected`: the less preferred answer

Instead of only teaching the model from ideal answers, DPO teaches the model to prefer better responses over weaker ones.

In this project, DPO is used to align the model toward my preferred study-assistant behavior:

> Prefer explanations that are technically correct, clear, concise, exam-ready, and useful for revision.

This makes the project more realistic than simple supervised fine-tuning, because the model is trained on comparative preferences rather than only single “correct” answers.


## Base Model

The base model used in this project is:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

This model was selected because it is small enough for local experimentation while still being instruction-tuned and capable of generating useful study explanations.

## Repository Structure

```text
.
├── configs/
│   ├── assistant_style.md
│   └── training_config.yaml
│
├── data/
│   ├── raw/
│   │   └── study_prompts.csv
│   ├── processed/
│   ├── preferences/
│   └── test/
│       └── heldout_prompts.csv
│
├── notebooks/
│
├── src/
│   ├── data/
│   │   ├── generate_candidates.py
│   │   ├── validate_candidates.py
│   │   ├── create_labeling_file.py
│   │   ├── build_dpo_dataset.py
│   │   └── validate_dpo_dataset.py
│   │
│   ├── training/
│   │   └── train_dpo.py
│   │
│   ├── evaluation/
│   │   ├── generate_eval_answers.py
│   │   ├── generate_dpo_answers.py
│   │   ├── create_eval_sheet.py
│   │   ├── create_dpo_comparison_sheet.py
│   │   └── summarize_eval_scores.py
│   │
│   └── app/
│
├── reports/
│   ├── baseline_evaluation.md
│   ├── dpo_tuning_summary.md
│   └── error_analysis.md
│
├── models/
│   └── README.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

### Prompt Dataset

The initial prompt dataset contains 100 study questions:

| Topic         | Number of Prompts |
| ------------- | ----------------: |
| Deep Learning |                60 |
| NLP           |                40 |

Each prompt includes:

* `id`
* `topic`
* `subtopic`
* `prompt`
* `difficulty`

Example:

```json
{
  "id": "dl_001",
  "topic": "deep_learning",
  "subtopic": "backpropagation",
  "prompt": "Explain backpropagation in simple language for exam revision.",
  "difficulty": "easy"
}
```

### Candidate Answer Dataset

For each prompt, two candidate answers were generated:

* `answer_a`: study-friendly, exam-oriented answer
* `answer_b`: more formal/generic answer

These candidate answers were saved as JSONL records and later used for preference labeling.

### Preference Dataset

Each candidate pair was manually labeled using the following options:

| Label  | Meaning                        |
| ------ | ------------------------------ |
| `A`    | `answer_a` is preferred        |
| `B`    | `answer_b` is preferred        |
| `skip` | pair is excluded from training |

The final DPO dataset uses only rows labeled `A` or `B`.

The final DPO training format is:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
```

A chat-style version of the DPO dataset was also created for better compatibility with the instruction-tuned Qwen model.

## Preference Labeling Rubric

The preferred answer should be more useful for a student preparing for an exam.

The labeling rubric prioritizes:

| Criterion             | Priority |
| --------------------- | -------- |
| Technical correctness | Highest  |
| Clarity               | High     |
| Exam usefulness       | High     |
| Structure             | Medium   |
| Conciseness           | Medium   |
| Example quality       | Medium   |

A friendly answer that is technically wrong should not be preferred over a formal but correct answer.

## Baseline Evaluation

Before DPO fine-tuning, the base model was evaluated on a held-out test set of 20 prompts:

| Topic         | Number of Prompts |
| ------------- | ----------------: |
| Deep Learning |                10 |
| NLP           |                10 |

The baseline answers were scored manually using a 1–5 rubric:

| Metric          | Description                                                        |
| --------------- | ------------------------------------------------------------------ |
| Clarity         | How easy the answer is to understand                               |
| Correctness     | Whether the answer is technically accurate                         |
| Exam usefulness | Whether the answer helps with revision or exam-style understanding |
| Conciseness     | Whether the answer is complete without unnecessary rambling        |
| Overall         | Overall answer quality                                             |

The baseline model performed strongly, which made improvement through DPO more challenging.

## DPO Fine-Tuning

The model was fine-tuned using:

* Base model: `Qwen/Qwen2.5-0.5B-Instruct`
* Fine-tuning method: LoRA
* Training method: DPO
* Dataset format: chat-style preference dataset
* Trainable parameter percentage: approximately 0.44%

### LoRA Configuration

```text
LoRA r: 16
LoRA alpha: 32
LoRA dropout: 0.05
Target modules:
- q_proj
- k_proj
- v_proj
- o_proj
```

### Initial DPO Run

The first DPO run successfully trained a LoRA adapter, but the post-DPO evaluation showed that it underperformed the base model. This suggested that the update may have shifted the model too aggressively given the small preference dataset.

## Hyperparameter Tuning

After the initial DPO run, a small tuning experiment was conducted.

The best DPO configuration was:

```text
learning_rate = 1e-6
beta = 0.1
max_steps = 20
LoRA r = 16
LoRA alpha = 32
LoRA dropout = 0.05
```

This tuned run was more stable and performed substantially better than the initial DPO run.

## Evaluation Results

The best tuned DPO model became competitive with the base model, but did not clearly outperform it on average.

| Metric          | Baseline Avg | Tuned DPO Avg | Change |
| --------------- | -----------: | ------------: | -----: |
| Clarity         |         4.60 |          4.35 |  -0.25 |
| Correctness     |         4.30 |          4.25 |  -0.05 |
| Exam usefulness |         4.00 |          4.05 |  +0.05 |
| Conciseness     |         4.40 |          4.25 |  -0.15 |
| Overall         |         4.33 |          4.23 |  -0.10 |

Winner counts across 20 held-out prompts:

| Winner   | Count |
| -------- | ----: |
| DPO      |     8 |
| Tie      |     6 |
| Baseline |     6 |

## Key Findings

The tuned DPO model improved compared with the first DPO run and outperformed the baseline on several prompts. However, the base model remained slightly stronger on average.

The strongest improvement was in **exam usefulness**, where the tuned DPO model slightly outperformed the baseline.

This suggests that preference tuning did influence the model toward the intended study-assistant behavior, but the small dataset size limited the overall improvement.

## Interpretation

This project shows that DPO is sensitive to both dataset quality and hyperparameters.

The first DPO run underperformed, while a gentler configuration improved stability and produced more competitive results. This indicates that when the base model is already strong and the preference dataset is small, aggressive fine-tuning can degrade performance.

The most important result is not that DPO automatically improved the model, but that the project demonstrates a realistic alignment workflow:

1. Build a preference dataset.
2. Fine-tune with DPO.
3. Evaluate against a baseline.
4. Identify failure modes.
5. Tune hyperparameters.
6. Compare results honestly.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Validate the prompt dataset

```bash
PYTHONPATH=. python src/data/validate_prompts.py
```

### 3. Generate candidate answers

```bash
PYTHONPATH=. python src/data/generate_candidates.py --limit 100
```

### 4. Validate candidate answers

```bash
PYTHONPATH=. python src/data/validate_candidates.py \
  --input-path data/processed/candidate_answers_100_sample.jsonl
```

### 5. Create labeling file

```bash
PYTHONPATH=. python src/data/create_labeling_file.py \
  --input-path data/processed/candidate_answers_100_sample.jsonl \
  --output-path data/preferences/preference_labels_100.csv
```

### 6. Build DPO dataset

```bash
PYTHONPATH=. python src/data/build_dpo_dataset.py \
  --input-path data/preferences/preference_labels_100.csv \
  --train-output data/preferences/dpo_train.jsonl \
  --eval-output data/preferences/dpo_eval.jsonl
```

### 7. Validate DPO dataset

```bash
PYTHONPATH=. python src/data/validate_dpo_dataset.py \
  --input-path data/preferences/dpo_train.jsonl

PYTHONPATH=. python src/data/validate_dpo_dataset.py \
  --input-path data/preferences/dpo_eval.jsonl
```

### 8. Train best DPO adapter

```bash
PYTHONPATH=. python src/training/train_dpo.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --train-path data/preferences/dpo_train_chat.jsonl \
  --eval-path data/preferences/dpo_eval_chat.jsonl \
  --output-dir models/tuning/dpo_lr1e-6_beta0.1_steps20_r16 \
  --max-steps 20 \
  --learning-rate 1e-6 \
  --beta 0.1 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

### 9. Generate DPO answers

```bash
PYTHONPATH=. python src/evaluation/generate_dpo_answers.py \
  --input-path data/test/heldout_prompts.csv \
  --output-path reports/tuning/dpo_answers_lr1e-6_beta0.1_steps20_r16.csv \
  --adapter-path models/tuning/dpo_lr1e-6_beta0.1_steps20_r16 \
  --limit 20
```

### 10. Create comparison sheet

```bash
PYTHONPATH=. python src/evaluation/create_dpo_comparison_sheet.py \
  --baseline-eval-path reports/baseline_evaluation.csv \
  --dpo-path reports/tuning/dpo_answers_lr1e-6_beta0.1_steps20_r16.csv \
  --output-path reports/tuning/dpo_comparison_lr1e-6_beta0.1_steps20_r16.csv
```

## Limitations

This project has several limitations:

* The preference dataset is small.
* The evaluation set contains only 20 held-out prompts.
* Scoring was done manually, so results may contain subjective judgment.
* Candidate answer quality varied, making some preference pairs weak.
* The base model was already strong, making improvement difficult.
* The final DPO model did not outperform the baseline on average.
* The project focuses only on Deep Learning and NLP study questions.

## Future Work

Possible improvements include:

* Increase the preference dataset beyond 100 pairs.
* Improve candidate answer generation to create stronger contrast pairs.
* Add more study domains such as cloud computing, cybersecurity, and statistics.
* Use multiple annotators or repeated scoring for more reliable evaluation.
* Try additional DPO hyperparameters and schedulers.
* Compare DPO with supervised fine-tuning.
* Deploy a Streamlit demo for side-by-side base vs DPO comparison.
* Upload the LoRA adapter to Hugging Face Hub.
* Add automated evaluation using an LLM judge as a secondary signal.

## Project Status

Current status:

```text
End-to-end MVP complete.
```

Completed components:

* Prompt dataset
* Candidate answer generation
* Preference labeling
* DPO dataset construction
* Baseline evaluation
* DPO fine-tuning
* Hyperparameter tuning
* Post-DPO comparison
* Final analysis reports

## Main Takeaway

This project demonstrates a realistic preference-tuning workflow for a study assistant.

The original goal was personal and practical: to build a model that explains difficult technical concepts in the way I actually prefer to study them — simple, structured, and exam-ready.

The tuned DPO model became more competitive after hyperparameter tuning and improved some held-out prompts, especially in exam usefulness. However, the base model remained slightly stronger overall.

This result highlights an important lesson:

> Preference tuning is not automatically better. It depends heavily on preference data quality, training setup, and evaluation discipline.

Even though the tuned model did not clearly outperform the base model on average, the project successfully demonstrates the full alignment workflow: dataset creation, preference labeling, DPO training, evaluation, hyperparameter tuning, and honest analysis.
