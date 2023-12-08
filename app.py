from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

app = Flask(__name__, static_folder='statics')

# Load the model
with open('song_recommendation_model.pkl', 'rb') as f:
    user_song_matrix, user_similarity_df = pickle.load(f)

def recommend_songs(user_id, song_title, user_song_matrix, user_similarity_df, num_recommendations=5):
    # Check if user exists
    if user_id not in user_song_matrix.index:
        return ["User not found in the dataset."]
    
    # Check if the song exists
    if song_title not in user_song_matrix.columns:
        return ["Song title not found in the dataset."]
    
    # Get top similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)  # remove the user itself
    top_users = similar_users.head(3)

    # Get songs listened by top similar users
    recommended_songs = []
    for similar_user in top_users.index:
        user_songs = user_song_matrix.loc[similar_user]
        user_songs = user_songs.apply(lambda x: x * top_users[similar_user])
        recommended_songs.append(user_songs)

    # Concatenate all series into one
    recommended_songs = pd.concat(recommended_songs, axis=0)

    recommended_songs = recommended_songs.groupby(recommended_songs.index).sum()

    # Remove songs already listened to by the user and the input song
    listened_songs = user_song_matrix.loc[user_id]
    recommended_songs = recommended_songs.drop(listened_songs[listened_songs > 0].index, errors='ignore')
    recommended_songs = recommended_songs.drop(song_title, errors='ignore')

    recommended_songs = recommended_songs.sort_values(ascending=False).head(num_recommendations)

    return recommended_songs.index.tolist()

@app.route('/')
def index():
    # Get a list of user IDs and song titles
    user_ids = user_song_matrix.index.tolist()
    song_titles = user_song_matrix.columns.tolist()

    return render_template('index.html', user_ids=user_ids, song_titles=song_titles)

@app.route('/most-listened-songs')
def most_listened_songs():
    return render_template('most_listened_songs.html')

@app.route('/top-artists')
def top_artists():
    return render_template('top_artists.html')

@app.route('/latest-hits')
def latest_hits():
    return render_template('latest_hits.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form['user_id']
    song_title = request.form['song_title']

    recommendations = recommend_songs(user_id, song_title, user_song_matrix, user_similarity_df, 5)

    return render_template('recommend.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
