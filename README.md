## Movie_recommendation

# Introduction
This technical document provides a step-by-step guide for building a movie recommendation system. The system will use collaborative filtering or matrix factorization techniques to recommend movies to users based on their past preferences and interactions. The document covers data acquisition, preprocessing, model development, evaluation, and fine-tuning.

# Dataset
The MovieLens dataset will be used for this recommendation system. It includes user ratings for movies and is widely used for movie recommendation tasks. The dataset can be downloaded from the MovieLens website (https://grouplens.org/datasets/movielens/).

# Step 1: Data Acquisition
Load and Preprocess the Data
Load the Raw Data: Download the MovieLens dataset and load it into a Pandas DataFrame. The dataset typically includes user IDs, movie IDs, ratings, and timestamps.
```shell
import pandas as pd
data_dir = 'path_to_dataset_directory/'
ratings_file = data_dir + 'u.data'
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_df = pd.read_csv(ratings_file, sep='\t', names=ratings_cols)
```

Preprocessing:
Drop Unnecessary Columns: Remove columns like 'timestamp' if not needed.
Handle Missing Values: Check and handle missing values (if any).
Convert Categorical Variables: Convert categorical variables like 'user_id' and 'movie_id' into numerical representations.

```shell
#converting categorical variables to numerical representations
ratings_df['user_id'] = ratings_df['user_id'].astype('category').cat.codes
ratings_df['movie_id'] = ratings_df['movie_id'].astype('category').cat.codes
```

# Step 2: Building the Recommendation System
Matrix Factorization (SVD)
Create a Surprise Dataset:
Use the Surprise library to create a Dataset object from the Pandas DataFrame.

```shell
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
reader = Reader(rating_scale=(1, 5))  
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
```

Split Data:
Split the dataset into training and testing sets for model evaluation.
```shell
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
```

Initialize and Train the Model:
Initialize the SVD model with suitable hyperparameters.
Fit the model on the training data.
```shell
svd_model = SVD(n_factors=100, n_epochs=20, random_state=42)
svd_model.fit(trainset)
```

# Recommendation Generation
Generate Recommendations:
Choose a user for whom you want to generate recommendations.
Identify movies the user has not yet rated.
Predict ratings for those unrated movies.
Sort and present the top N recommended movies.

```shell
user_id_to_predict = 0  # Change to the user of interest
user_ratings = ratings_df[ratings_df['user_id'] == user_id_to_predict]
movies_not_rated_by_user = ratings_df[~ratings_df['movie_id'].isin(user_ratings['movie_id'])]

# Generate predictions for unrated movies
predictions = [svd_model.predict(user_id_to_predict, movie_id) for movie_id in movies_not_rated_by_user['movie_id']]

# Sort predictions by estimated rating (higher first)
sorted_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

# Get the top N recommended movie IDs
top_n = 10  # Change to the desired number of recommendations
top_movie_ids = [prediction.iid for prediction in sorted_predictions[:top_n]]
```

# Step 3: Evaluation
Evaluate the Model:
Optionally, evaluate the model's performance on the test set using metrics like Root Mean Squared Error (RMSE).

```shell
test_predictions = svd_model.test(testset)
rmse = accuracy.rmse(test_predictions)
print("Root Mean Squared Error (RMSE) on test set: {:.4f}".format(rmse))
```

