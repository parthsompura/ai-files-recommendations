# -*- coding: utf-8 -*-
"""
@author: Parth Pathak
"""

# Library imports
import uvicorn
from fastapi import FastAPI

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

# Create the app object
app = FastAPI()

df = pd.read_csv("data-set.csv")
df = df.dropna()

df = df.reset_index(drop=True)


def duplicate_rows_randomly(df, n_duplicates):
    duplicated_rows = []

    for user_id in df['UserId'].unique():
        user_data = df[df['UserId'] == user_id]

        for _ in range(n_duplicates):
            random_row = user_data.sample(n=1).iloc[0]
            duplicated_rows.append(random_row)

    duplicated_df = pd.DataFrame(duplicated_rows)
    new_df = pd.concat([df, duplicated_df], ignore_index=True)
    return new_df


# Example usage:
# Assuming your DataFrame has columns 'UserID', 'FilePath', and 'Action'
# You can adjust n_duplicates based on how many times you want to duplicate each row
new_df = duplicate_rows_randomly(df, n_duplicates=100)

df = new_df

# Creating Features
label_encoder = LabelEncoder()
df['Actions'] = label_encoder.fit_transform(df['Actions'])

df['Rating'] = df.groupby(['UserId', 'FilePath'])['FilePath'].transform('count')
df["Rating"].unique()

# Content-Based Filtering using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['FilePath'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Collaborative Filtering using Surprise library
reader = Reader(rating_scale=(1, 10))
data_surprise = Dataset.load_from_df(df[['UserId', 'FilePath', 'Rating']], reader)
trainset, testset = train_test_split(data_surprise, test_size=0.2, random_state=42)

algo = SVD()
algo.fit(trainset)


# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World!'}


# Route with a single parameter, returns the parameter within a message
# Located at: http://127.0.0.1:8000/{name}
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome': f'{name}'}


# Route with a single parameter user_id, returns the files recommendations
# Located at: http://127.0.0.1:8000/v1/user/recommendation/{user_id}
@app.get('/v1/user/recommendation/{user_id}/{file_path}')
def get_recommendations(user_id: int, file_path: str | None = None):
    # Content-Based Filtering
    # df = df.reset_index()

    res = df[df['FilePath'].str.contains(file_path, case=False, regex=True)].index[0]

    sim_scores = list(enumerate(cosine_sim[res]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    excluded_file_path = df['FilePath'].iloc[res]
    sim_scores = [i for i in sim_scores if df['FilePath'].iloc[i[0]] != excluded_file_path]

    # sim_scores = sim_scores[:5]  # Top 5 similar files excluding the user_given_file_path
    # Select top unique 5 similar files
    unique_similar_files = set()
    top_similar_files = []
    for i in sim_scores:
        if df['FilePath'].iloc[i[0]] not in unique_similar_files:
            unique_similar_files.add(df['FilePath'].iloc[i[0]])
            top_similar_files.append(i)
        if len(top_similar_files) == 5:
            break

    file_indices = [i[0] for i in top_similar_files]

    content_based_recommendations = df['FilePath'].iloc[file_indices].tolist()

    collab_based_recommendations = []
    for item in df['FilePath'].unique():
        predicted_rating = algo.predict(user_id, item).est
        if predicted_rating > 7.0:
            collab_based_recommendations.append(item)

    return collab_based_recommendations + content_based_recommendations

    # return {f'Recommendations for {user_id}': f'{content_based_recommendations, collab_based_recommendations}'}
    # return collab_based_recommendations


# Run the API with uvicorn
# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn main:app --reload
