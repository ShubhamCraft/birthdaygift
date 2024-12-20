import os

from flask import Flask, render_template, request
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load CSV file
df = pd.read_csv("data/One_Direction_songs.csv")

# Preprocess data
themes_keywords = {
    "Love": "love heart kiss feel",
    "Adventure": "road fly dream journey sky",
    "Memories": "remember time past memory night",
}

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Lyrics'].fillna(''))

@app.route("/", methods=["GET", "POST"])
def home():
    theme_message = None
    year_results = pd.DataFrame()

    if request.method == "POST":
        if "theme" in request.form:
            theme = request.form["theme"]
            theme_query = themes_keywords.get(theme, "")

            # Calculate similarity
            query_vector = vectorizer.transform([theme_query])
            similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

            # Pick the most similar song
            best_match_idx = similarity_scores.argmax()
            if similarity_scores[best_match_idx] > 0:
                song = df.iloc[best_match_idx]
                theme_message = {
                    "song_title": song["Song"],
                    "album": song["Album(s)"],
                    "lyrics": song["Lyrics"],
                    "youtube_link": f"https://www.youtube.com/results?search_query={song['Song'].replace(' ', '+')}+One+Direction"
                }

        if "year" in request.form:
            selected_year = request.form["year"]
            year_results = df[df["Year"].astype(str) == selected_year]

    return render_template("index.html", theme_message=theme_message, year_results=year_results, years=sorted(df['Year'].dropna().unique()))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
