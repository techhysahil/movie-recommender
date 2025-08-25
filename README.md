# 🎬 Movie Recommender — Exploring AI Basics

## 📌 Motivation
The goal of this project is to **deep dive into AI fundamentals** — starting with how a **neuron** works, how multiple neurons form a **neural network**, and how such networks can be trained to solve real-world problems.

A recommender system is a practical example of this: it takes **user data** (age, gender, preferences) and **movie data** (genre, popularity, year) and predicts how much a user might **like a movie**. This connects theory (neurons, weights, activations, loss, optimization) with practice (predicting ratings).

---

## 🚀 What this project does
- Generates a **synthetic but realistic dataset** of users, movies, and ratings
- Builds a **neural network** using TensorFlow/Keras
- Trains the model to predict ratings (1–5)
- Shows **training progress** with loss/MAE curves
- Provides **EDA plots**:
    - Rating distribution
    - Age vs rating
    - Average rating by genre
- Includes **interactive sliders** (via ipywidgets) to test new user–movie pairs

---

## 🛠️ Tech Stack
- **Python 3**
- **TensorFlow / Keras** for building & training the neural network
- **pandas, numpy, scikit-learn** for data handling and preprocessing
- **matplotlib** for visualization
- **ipywidgets** for interactive demo inside Jupyter Notebook

---

## 📂 Files
- `Movie_Recommender_Notebook.ipynb` — full end-to-end workflow
- `users.csv`, `movies.csv`, `ratings.csv` — generated datasets
- `models/movie_model_with_prefs.keras` — trained model

---

## ▶️ How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib ipywidgets
