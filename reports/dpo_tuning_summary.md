# DPO Hyperparameter Tuning Summary

## Goal

The goal of this tuning phase was to improve the first DPO-tuned model, which underperformed the base model during post-training evaluation.

## Compared Runs

| Run | Learning Rate | Beta | Steps | LoRA r | Result |
|---|---:|---:|---:|---:|---|
| Initial DPO | 5e-6 | 0.1 | 30 | 16 | Underperformed baseline clearly |
| Tuned Run 1 | 1e-6 | 0.1 | 20 | 16 | Best result; near-baseline and improved several prompts |
| Tuned Run 2 | 5e-7 | 0.05 | 15 | 16 | Too conservative; weaker than Tuned Run 1 |

## Best Run

The best configuration was:

- Learning rate: 1e-6
- Beta: 0.1
- Max steps: 20
- LoRA r: 16
- LoRA alpha: 32
- LoRA dropout: 0.05

## Key Finding

The first DPO run likely shifted the model too aggressively. Reducing the learning rate and number of training steps improved stability and produced a more competitive model.

The best DPO run did not clearly outperform the base model on average, but it reduced the performance gap and improved several held-out prompts.

## Interpretation

Because the baseline model was already strong and the preference dataset was small, DPO improvements were limited. The experiment shows that preference tuning is sensitive to hyperparameters and dataset quality.

## Future Improvements

- Increase preference dataset size beyond 100 pairs.
- Improve chosen/rejected contrast quality.
- Add more consistent preference labeling.
- Try additional DPO settings such as constant learning rate scheduler.
- Evaluate with more held-out prompts.