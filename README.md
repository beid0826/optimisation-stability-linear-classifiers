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

## Key findings

- **Scale invariance:** Positively rescaling (θ, θ₀) leaves the classifier and decision boundary unchanged, but changes the signed agreements and gradients. The geometric margin stays the same.
- **Learning rate:** On standardised synthetic data (λ = 0.1), the learning rate mainly changes the speed of descent. None of the rates tested (0.001 to 1) diverged.
- **Feature scaling:** Multiplying one feature by 100 (training std ≈ 1.78 vs 173.14) made training very sensitive to the learning rate, with large oscillations. Standardising removed the problem.
- **Regularisation:** Larger λ keeps the weight norm smaller and makes training settle at a more strongly regularised solution.
- **Real data:** On Breast Cancer Wisconsin, raw features gave a large, oscillating objective, while standardised features gave smooth descent.

## How to run

pip install numpy matplotlib scikit-learn
python dissertation_code.py

The script generates all the figures as PDFs in the current folder. Each plot opens in a window, so close it to continue to the next one.
