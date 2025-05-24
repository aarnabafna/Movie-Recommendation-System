import streamlit as st
import pandas as pd
import difflib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# TMDB API Key
API_KEY = '43a476b1b6a29938820e3eac8a0f423e'

# --- Function to fetch movie details (poster, rating, overview) ---
def fetch_movie_details(movie_title):
    try:
        query = movie_title.replace(' ', '%20')
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
        response = requests.get(url)
        data = response.json()
        if data['results']:
            movie_data = data['results'][0]
            poster_path = movie_data.get('poster_path')
            full_poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            rating = movie_data.get('vote_average', 'N/A')
            overview = movie_data.get('overview', 'No description available.')
            return full_poster_path, rating, overview
        else:
            return None, 'N/A', 'No description available.'
    except:
        return None, 'N/A', 'No description available.'

# --- Data Collection and Pre-Processing ---
@st.cache_data
def load_data():
    movies_data = pd.read_csv('movies.csv.csv')
    
    # Handling missing values
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
        
    # Combining features for similarity
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
    
    # Vectorizing the features using TF-IDF
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    
    # Calculating similarity
    similarity = cosine_similarity(feature_vectors)
    
    return movies_data, similarity

# --- Load Data ---
movies_data, similarity = load_data()

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System")

# List of all movie titles
list_of_all_titles = movies_data['title'].tolist()

# Dropdown with search
selected_movie = st.selectbox('🔎 Search or select a movie:', options=[""] + list_of_all_titles)

# Centered "Recommend" button
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    recommend_clicked = st.button("Recommend")

# Button functionality
if recommend_clicked:
    
    if not selected_movie:
        st.warning("⚠️ Please select a movie from the dropdown.")
    
    elif selected_movie not in list_of_all_titles:
        st.error("❌ No search results found. Please try another movie.")
    
    else:
        st.write(f"Using movie: **{selected_movie}**")
        
        index_of_the_movie = movies_data[movies_data.title == selected_movie]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        top_movies = sorted_similar_movies[1:11]  # Exclude the movie itself
        movie_titles = [movies_data.iloc[movie[0]]['title'] for movie in top_movies]
        similarities = [movie[1] for movie in top_movies]
        
        # --- Display similar movies ---
        st.subheader('🎯 Movies Suggested for You:')
        
        for idx, title in enumerate(movie_titles):
            poster_url, rating, overview = fetch_movie_details(title)
            with st.container():
                st.markdown(f"### {idx+1}. {title}")
                cols = st.columns([1, 4])
                
                with cols[0]:
                    if poster_url:
                        st.image(poster_url, width=120)
                    else:
                        st.image('https://via.placeholder.com/120x180?text=No+Image', width=120)
                
                with cols[1]:
                    st.markdown(f"**Rating:** {rating} ⭐")
                    st.markdown(f"**Description:** {overview}")
                    st.markdown("---")

        # --- Plot the top 10 similar movies (Cosine Similarity Score Bar Chart) ---
        st.subheader(f"📊 Top 10 Movies Similar to '{selected_movie}'")
        plt.figure(figsize=(10, 5))
        plt.barh(movie_titles[::-1], similarities[::-1], color='skyblue')
        plt.xlabel("Cosine Similarity Score")
        plt.title(f"Top 10 Movies Similar to '{selected_movie}'")
        plt.tight_layout()
        st.pyplot(plt)
