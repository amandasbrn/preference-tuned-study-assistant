# Preference-Tuned Study Assistant using DPO

## Overview

This project aims to build a preference-tuned AI study assistant designed to produce clearer, more exam-oriented explanations for technical concepts.

The project uses Direct Preference Optimization (DPO) to fine-tune a small open-source language model using chosen/rejected answer pairs.

## Problem Statement

General-purpose LLMs can answer study questions, but their responses may not match a student's preferred learning style. They may be too verbose, too technical, or not structured for exam revision.

This project explores whether preference tuning can improve the usefulness of study explanations.

## Target Behaviour

The assistant should produce answers that are:

- Simple
- Exam-oriented
- Step-by-step
- Concise but complete
- Supported by examples where useful
- Technically accurate

## MVP Scope

This first version focuses on:

- Deep Learning concepts
- NLP concepts

## Method

1. Build a dataset of study prompts
2. Generate candidate answers
3. Label chosen/rejected preference pairs
4. Fine-tune a base model using DPO
5. Evaluate base model vs DPO-tuned model
6. Build a Streamlit demo

## Planned Model

Base model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The model will be evaluated using a rubric:

Criterion	Description
Clarity	Is the explanation easy to understand?
Exam usefulness	Is it useful for revision or exam answers?
Step-by-step explanation	Does it explain the reasoning clearly?
Correctness	Is the answer technically accurate?
Conciseness	Is it complete without being unnecessarily long?
Expected Output

The final project will include:

Preference dataset
DPO training script
Fine-tuned LoRA adapter
Evaluation report
Streamlit demo
Error analysis