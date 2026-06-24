# Study Prompt Dataset Card

## Dataset Name

LearnAlign Study Prompts

## Purpose

This dataset contains study questions used to build a preference-tuned AI study assistant. The prompts focus on technical concepts that students commonly ask about during exam revision.

## Scope

The MVP version focuses on two topics:

- Deep Learning
- Natural Language Processing

## Dataset Size

Total prompts: 100

Topic split:

- Deep Learning: 60 prompts
- NLP: 40 prompts

## Columns

| Column | Description |
|---|---|
| id | Unique prompt identifier |
| topic | Main topic area |
| subtopic | More specific concept category |
| prompt | Study question given to the model |
| difficulty | Estimated difficulty: easy, medium, or hard |

## Target Assistant Behaviour

The assistant should answer these prompts in a way that is:

- Simple
- Beginner-friendly
- Exam-oriented
- Step-by-step
- Concise but complete
- Technically accurate

## Use in Project

This prompt dataset will be used in later phases to:

1. Generate multiple candidate answers
2. Create chosen/rejected preference pairs
3. Build a DPO training dataset
4. Evaluate the base model and the preference-tuned model