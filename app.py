import streamlit as st
from src.recommender import recommend, movie_matrix

st.title("🎬 Movie Recommendation System")

movie = st.selectbox(
    "Select a movie",
    movie_matrix.columns
)

if st.button("Recommend"):

    st.write("### Recommended Movies:")

    movie_index = movie_matrix.columns.get_loc(movie)
    
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_scores = cosine_similarity(movie_matrix.T)

    scores = list(enumerate(similarity_scores[movie_index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    for i in sorted_scores[1:6]:
        st.write(movie_matrix.columns[i[0]])