# Mean-Field-Type Transformers

Research and implementation project exploring Transformer architectures and distributional mean-field-type ideas, developed as my undergraduate final-year project in Computer Science.

## Overview

This project studies attention-based neural networks and implements a Transformer-inspired encoder in **PyTorch** for a binary classification task using Formula 1 racing data.

The work explores:
- Self-attention and multi-head attention
- Positional encoding and residual connections
- Transformer encoder blocks
- Different activation functions: ReLU, GELU, Tanh, Sigmoid and LeakyReLU
- Distributional mean-field-type Transformer concepts

## Dataset and task

The classification target is whether a driver wins a race. The project uses numerical features related to drivers, constructors and race results, including driver ID, constructor ID, race ID, grid position, laps and points.

The data was split into training and test sets (80/20).

> **Modeling note:** Features such as final points and laps may contain information not available before a race. Any results should therefore be interpreted as a classification experiment rather than a validated pre-race forecasting system. Identifier columns also require careful treatment.

## Model configuration

| Parameter | Value |
| --- | --- |
| Framework | PyTorch |
| Model dimension | 6 |
| Feed-forward dimension | 12 |
| Encoder layers | 2 |
| Attention heads | 2 |
| Batch size | 64 |
| Epochs | 10 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Binary cross-entropy |

## Project context

Bachelor's final-year project (2025), Sciences Mathématiques et Informatique, Université Ibn Tofaïl, Kénitra, Morocco.

This repository documents an academic learning and research implementation. It should not be interpreted as evidence that a production-grade predictive system has been validated.

## Next steps

- Document the exact environment and reproducible training commands
- Add evaluation metrics and plots with a clear test protocol
- Evaluate data leakage and class imbalance
- Clarify which components implement the mean-field-type formulation versus a standard Transformer encoder

## Author

**Hidaya Kettani** — Computer Science graduate and MSc student in Artificial Intelligence for Connected Industries at Universitat Politècnica de Catalunya.

[LinkedIn](https://www.linkedin.com/in/hidaya-kettani-9a17b2334/) · [GitHub](https://github.com/hidayakettani)
