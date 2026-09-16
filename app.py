
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")
st.title("🧳 Tourism Experience Analytics")
st.caption("Classification, Prediction, and Recommendation System")

@st.cache_data
def load_data():
    destinations_df = pd.read_csv("dataset/Expanded_Destinations.csv")
    userhistory_df = pd.read_csv("dataset/Final_Updated_Expanded_UserHistory.csv")
    return destinations_df, userhistory_df


@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    model_columns = pickle.load(open("model_columns.pkl", "rb"))
    return model, model_columns


@st.cache_resource
def build_similarity(destinations_df):
    destinations_df = destinations_df.copy()
    destinations_df["content"] = (
        destinations_df["Type"].astype(str) + " " + destinations_df["State"].astype(str)
    )
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(destinations_df["content"])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


destinations_df, userhistory_df = load_data()
cosine_sim = build_similarity(destinations_df)

try:
    model, model_columns = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ----------------------------------------------------------------
# Recommendation function (content-based filtering)
# ----------------------------------------------------------------
def recommend_destinations(user_id, userhistory_df, destinations_df, cosine_sim, top_n=5):
    id_to_idx = pd.Series(destinations_df.index, index=destinations_df["DestinationID"]).to_dict()
    visited_destinations = userhistory_df[
        userhistory_df["UserID"] == user_id
    ]["DestinationID"].values

    if len(visited_destinations) == 0:
        return pd.DataFrame(), "No visit history found for this UserID."

    visited_idx = [id_to_idx[d] for d in visited_destinations if d in id_to_idx]
    if len(visited_idx) == 0:
        return pd.DataFrame(), "Visited destinations not found in the destinations table."

    similar_scores = np.sum(cosine_sim[visited_idx], axis=0)
    recommended_idx = np.argsort(similar_scores)[::-1]

    recommendations = []
    for idx in recommended_idx:
        dest_id = destinations_df.iloc[idx]["DestinationID"]
        if dest_id not in visited_destinations:
            recommendations.append(destinations_df.iloc[idx][[
                "DestinationID", "Name", "State", "Type", "Popularity", "BestTimeToVisit"
            ]].to_dict())
            if len(recommendations) >= top_n:
                break

    return pd.DataFrame(recommendations), None


# ----------------------------------------------------------------
# Popularity/rating prediction function (matches one-hot training)
# ----------------------------------------------------------------
def predict_popularity(user_input, model, model_columns):
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(
        input_df, columns=["State", "Type", "BestTimeToVisit", "Preferences", "Gender"]
    )
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    return model.predict(input_df)[0]


# ----------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Insights", "⭐ Predict Popularity", "🎯 Recommendations"])

# ---------------- Tab 1: Insights ----------------
with tab1:
    st.header("Tourism Trends")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Most Popular Destinations")
        top_dest = destinations_df.sort_values("Popularity", ascending=False).head(10)
        st.bar_chart(top_dest.set_index("Name")["Popularity"])

    with col2:
        st.subheader("Destinations by Type")
        st.bar_chart(destinations_df["Type"].value_counts())

    st.subheader("Destinations by State")
    st.bar_chart(destinations_df["State"].value_counts())

# ---------------- Tab 2: Predict Popularity ----------------
with tab2:
    st.header("Predict Destination Popularity Score")

    if not model_loaded:
        st.warning(
            "model.pkl and/or model_columns.pkl not found in this folder. "
            "Train and save your model first (see your notebook), then place "
            "both files next to app.py."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            dest_type = st.selectbox("Destination Type", sorted(destinations_df["Type"].unique()))
            state = st.selectbox("State", sorted(destinations_df["State"].unique()))
            best_time = st.selectbox(
                "Best Time to Visit", sorted(destinations_df["BestTimeToVisit"].unique())
            )
        with col2:
            preferences = st.text_input("User Preferences (e.g. 'City, Historical')", "City, Historical")
            gender = st.selectbox("Gender", ["Male", "Female"])
            num_adults = st.number_input("Number of Adults", min_value=1, value=2)
            num_children = st.number_input("Number of Children", min_value=0, value=0)

        if st.button("Predict Popularity"):
            user_input = {
                "Type": dest_type,
                "State": state,
                "BestTimeToVisit": best_time,
                "Preferences": preferences,
                "Gender": gender,
                "NumberOfAdults": num_adults,
                "NumberOfChildren": num_children,
            }
            score = predict_popularity(user_input, model, model_columns)
            st.success(f"Predicted Popularity Score: {score:.2f}")

# ---------------- Tab 3: Recommendations ----------------
with tab3:
    st.header("Get Personalized Destination Recommendations")

    user_ids = sorted(userhistory_df["UserID"].unique())
    selected_user = st.selectbox("Select a User ID", user_ids)

    if st.button("Get Recommendations"):
        recs, error = recommend_destinations(selected_user, userhistory_df, destinations_df, cosine_sim)
        if error:
            st.warning(error)
        else:
            st.dataframe(recs, use_container_width=True)
