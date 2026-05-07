# Optimisation and Stability in Linear Classification Models

This repository contains code and selected outputs from my final-year mathematics dissertation at the University of Southampton.

The project studies how optimisation choices affect the training behaviour of linear binary classifiers. In particular, it investigates the effects of learning rate, L2 regularisation and feature scaling on convergence, objective value, weight norm and decision-boundary behaviour.

## Project overview

Linear classifiers are mathematically transparent models, but their training behaviour can still be sensitive to optimisation choices. This project uses hinge loss with L2 regularisation and gradient/subgradient descent to study how the same classification problem can lead to different optimisation paths under different learning rates, scaling choices and regularisation strengths.

## Main topics

- Linear binary classification
- Hinge loss
- Regularised empirical risk minimisation
- Gradient and subgradient descent
- Learning-rate stability
- L2 regularisation
- Feature standardisation
- Loss and weight-norm diagnostics

## Repository structure

- `src/`: Python scripts for optimisation, experiments and plotting
- `notebooks/`: exploratory notebooks
- `figures/`: selected figures from the experiments
- `report/`: short project summary

## Tools used

Python, NumPy, scikit-learn, Matplotlib, LaTeX
