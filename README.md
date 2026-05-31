# Movie Recommendation System

A hybrid movie recommendation system built on the MovieLens 1M dataset. The project compares three models and combines collaborative filtering with content-based filtering to address sparsity and the cold start problem.

BSc Project — Computer Science, AI and Data Science specialisation.

## Overview

The system predicts the rating a user would give to a film they have not seen, then ranks films by predicted rating to produce recommendations. This is a regression task on a 1 to 5 scale.

Two problems shape the design:

- **Sparsity** — about 95.5% of the user-item matrix is empty.
- **Cold start** — new users have no rating history for collaborative filtering to use.

## Models

| Model | Type | Role |
|-------|------|------|
| User-Based CF (Pearson) | Memory-based collaborative filtering | Baseline / reference point |
| SVD | Model-based collaborative filtering (matrix factorisation) | Main ML model, handles sparsity |
| Hybrid (SVD + CBF) | SVD + TF-IDF content-based filtering | Final model, handles cold start |

The hybrid blends the two components with a weight `alpha` that depends on how many ratings a user has. New users rely on content-based filtering (`alpha = 0`); experienced users rely on SVD (`alpha = 1`).

## Results

| Model | RMSE | MAE |
|-------|------|-----|
| Baseline (User-Based CF) | 0.9604 | 0.7642 |
| SVD | 0.8729 | 0.6845 |
| Hybrid (SVD + CBF) | 0.8696 | 0.6822 |

Cold start (users with fewer than 25 ratings): the hybrid reduces RMSE from 0.9684 to 0.9011, a 7% improvement over SVD, while matching SVD exactly for experienced users.

## Project Structure

```
movie-recommendation-system/
  ml-1m/                  # MovieLens 1M dataset (not tracked, download separately)
  phase2/
    phase2.py             # EDA and baseline model
  phase3.ipynb            # SVD, hybrid, model comparison, learning curve
  phase4.ipynb            # error analysis and model interpretation
  README.md
```

## Setup

Python 3.11 is required (scikit-surprise does not yet support 3.14).

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-surprise scikit-learn
pip install "numpy<2"              # surprise needs NumPy 1.x
```

Download the dataset and place it so the files sit at `ml-1m/ratings.dat`, `ml-1m/movies.dat`, `ml-1m/users.dat`:

https://files.grouplens.org/datasets/movielens/ml-1m.zip

## Running

```bash
# Phase 2: EDA and baseline
python phase2/phase2.py

# Phase 3 and 4: open in Jupyter and run all cells
jupyter notebook
```

## Evaluation

All models use the same 80/20 train/test split with a fixed random seed for fair comparison. SVD is additionally validated with 5-fold cross-validation. Metrics are RMSE and MAE. The test set is used only once, after model selection, to avoid data leakage.

## Dataset

MovieLens 1M — 1,000,209 ratings, 6,040 users, 3,706 films. Published by GroupLens Research.
