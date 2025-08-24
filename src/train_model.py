import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

# --- Step 1: Simulate realistic user data ---
np.random.seed(42)

num_users = 50
users = pd.DataFrame({
    'user_id': np.arange(1, num_users+1),
    'age': np.random.randint(18, 60, size=num_users),
    'gender': np.random.randint(0, 2, size=num_users),  # 0=female,1=male
    'pref_action': np.random.rand(num_users),
    'pref_comedy': np.random.rand(num_users),
    'pref_drama': np.random.rand(num_users)
})

# --- Step 2: Simulate movies dataset ---
num_movies = 50
movies = pd.DataFrame({
    'movie_id': np.arange(101, 101+num_movies),
    'genre': np.random.randint(0, 3, size=num_movies),  # 0=Action,1=Comedy,2=Drama
    'popularity': np.random.randint(50, 100, size=num_movies),
    'year': np.random.randint(1980, 2024, size=num_movies)
})

# --- Step 3: Simulate ratings ---
ratings_list = []
for _, row in users.iterrows():
    movie_choices = np.random.choice(movies['movie_id'], 30, replace=False)  # 30 movies per user
    for movie_id in movie_choices:
        movie_genre = movies.loc[movies['movie_id']==movie_id, 'genre'].values[0]
        if movie_genre == 0:
            rating = int(np.clip(1 + 4*row['pref_action'] + np.random.randn()*0.2, 1, 5))
        elif movie_genre == 1:
            rating = int(np.clip(1 + 4*row['pref_comedy'] + np.random.randn()*0.2, 1, 5))
        else:
            rating = int(np.clip(1 + 4*row['pref_drama'] + np.random.randn()*0.2, 1, 5))
        ratings_list.append([row['user_id'], movie_id, rating])

ratings = pd.DataFrame(ratings_list, columns=['user_id', 'movie_id', 'rating'])

# --- Step 4: Preprocess ---
# Include user preference columns in features
data = ratings.merge(users, on='user_id').merge(movies, on='movie_id')
X = data[['age', 'gender', 'genre', 'popularity', 'year',
          'pref_action', 'pref_comedy', 'pref_drama']].values
y = data['rating'].values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 5: Build model ---
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(8,), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# --- Step 6: Chunked training for visible progress ---
epochs_total = 300
chunk_size = 50

for i in range(0, epochs_total, chunk_size):
    print(f"\nTraining epochs {i+1} to {i+chunk_size}")
    model.fit(X_scaled, y, epochs=chunk_size, verbose=2)

# --- Step 7: Save model automatically ---
os.makedirs("models", exist_ok=True)
model_path = "models/movie_model_with_prefs.keras"
model.save(model_path)
print(f"\nModel saved to {model_path}")

# --- Step 8: Make predictions ---
# Example: 27-year-old female, Comedy movie released 2015, popularity 80, with preference for Comedy
new_data = np.array([[27, 0, 1, 80, 2015, 0.2, 0.9, 0.3]])  # pref_action=0.2, pref_comedy=0.9, pref_drama=0.3
new_data_scaled = scaler.transform(new_data)
predicted_rating = model.predict(new_data_scaled)
print("\nPredicted rating for new user-movie pair:", predicted_rating[0][0])
