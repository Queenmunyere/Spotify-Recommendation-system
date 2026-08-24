"""
Spotify Audio Recommendation System
Content-based recommender using audio features (danceability, energy, valence, etc.)
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# Page config & style
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spotify Audio Recommendation System",
    page_icon=":musical_note:",
    layout="wide",
)

ACCENT = "#1B3A5C"      # navy
ACCENT_2 = "#8A6D1D"    # gold
GOOD = "#1E6B45"        # green

sns.set_theme(style="whitegrid")
plt.rcParams["axes.titlecolor"] = ACCENT
plt.rcParams["axes.titleweight"] = "bold"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FAFAFA; }}
    h1, h2, h3 {{ color: {ACCENT}; }}
    div.stButton > button {{
        background-color: {ACCENT};
        color: white;
        border: none;
    }}
    div.stButton > button:hover {{
        background-color: {ACCENT_2};
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
    "key", "mode", "time_signature",
]

REQUIRED_COLUMNS = FEATURES + ["liked"]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def make_sample_dataset(n=200, seed=7):
    """Learner-friendly synthetic dataset with the same schema as the
    Kaggle Spotify recommendation dataset, used when no CSV is uploaded."""
    rng = np.random.default_rng(seed)
    n_liked = n // 2
    n_disliked = n - n_liked

    liked = pd.DataFrame({
        "danceability": rng.normal(0.78, 0.08, n_liked).clip(0, 1),
        "energy": rng.normal(0.72, 0.12, n_liked).clip(0, 1),
        "key": rng.integers(0, 12, n_liked),
        "loudness": rng.normal(-5.5, 2.0, n_liked),
        "mode": rng.integers(0, 2, n_liked),
        "speechiness": rng.normal(0.08, 0.05, n_liked).clip(0, 1),
        "acousticness": rng.normal(0.15, 0.12, n_liked).clip(0, 1),
        "instrumentalness": rng.exponential(0.03, n_liked).clip(0, 1),
        "liveness": rng.normal(0.18, 0.1, n_liked).clip(0, 1),
        "valence": rng.normal(0.68, 0.15, n_liked).clip(0, 1),
        "tempo": rng.normal(120, 18, n_liked).clip(60, 200),
        "duration_ms": rng.normal(210000, 25000, n_liked).clip(90000, 350000),
        "time_signature": rng.choice([3, 4, 4, 4, 5], n_liked),
        "liked": 1,
    })

    disliked = pd.DataFrame({
        "danceability": rng.normal(0.35, 0.15, n_disliked).clip(0, 1),
        "energy": rng.normal(0.35, 0.18, n_disliked).clip(0, 1),
        "key": rng.integers(0, 12, n_disliked),
        "loudness": rng.normal(-11.0, 4.0, n_disliked),
        "mode": rng.integers(0, 2, n_disliked),
        "speechiness": rng.normal(0.06, 0.06, n_disliked).clip(0, 1),
        "acousticness": rng.normal(0.5, 0.28, n_disliked).clip(0, 1),
        "instrumentalness": rng.exponential(0.15, n_disliked).clip(0, 1),
        "liveness": rng.normal(0.2, 0.13, n_disliked).clip(0, 1),
        "valence": rng.normal(0.32, 0.16, n_disliked).clip(0, 1),
        "tempo": rng.normal(95, 22, n_disliked).clip(50, 190),
        "duration_ms": rng.normal(230000, 40000, n_disliked).clip(90000, 400000),
        "time_signature": rng.choice([3, 4, 4, 4, 5], n_disliked),
        "liked": 0,
    })

    df = pd.concat([liked, disliked], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "track_id", [f"track_{i:03d}" for i in range(len(df))])
    return df


def load_uploaded(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded file is missing required column(s): {', '.join(missing)}"
        )
    return df


@st.cache_data
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    return df


@st.cache_data
def build_similarity(df: pd.DataFrame):
    X = df[FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sim_matrix = cosine_similarity(X_scaled)
    return X_scaled, sim_matrix


def track_label(df: pd.DataFrame, idx: int) -> str:
    if "track_id" in df.columns:
        return f"{idx} — {df.loc[idx, 'track_id']}"
    if "name" in df.columns:
        return f"{idx} — {df.loc[idx, 'name']}"
    return f"Track {idx}"


def recommend_from_song(sim_matrix, df, song_index, n=5):
    scores = list(enumerate(sim_matrix[song_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != song_index][:n]
    idx = [i for i, _ in scores]
    result = df.iloc[idx].copy()
    result["similarity_score"] = [round(s, 4) for _, s in scores]
    return result


def recommend_from_profile(X_scaled, df, n=10):
    liked_idx = df.index[df["liked"] == 1].tolist()
    if not liked_idx:
        return pd.DataFrame(), None
    profile = X_scaled[liked_idx].mean(axis=0)
    scores = cosine_similarity([profile], X_scaled)[0]
    order = np.argsort(scores)[::-1]
    order = [i for i in order if i not in liked_idx][:n]
    result = df.iloc[order].copy()
    result["similarity_score"] = [round(scores[i], 4) for i in order]
    return result, profile


# ----------------------------------------------------------------------------
# Sidebar — data source
# ----------------------------------------------------------------------------
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Spotify CSV", type=["csv"])
st.sidebar.caption(
    "Expected columns: " + ", ".join(REQUIRED_COLUMNS)
)

if uploaded_file is not None:
    try:
        raw_df = load_uploaded(uploaded_file)
        source_note = f"Loaded `{uploaded_file.name}` ({len(raw_df)} tracks)"
    except ValueError as e:
        st.sidebar.error(str(e))
        st.stop()
else:
    raw_df = make_sample_dataset()
    source_note = "Using a generated sample dataset (upload a CSV to use your own)"

st.sidebar.info(source_note)

df = clean(raw_df)
X_scaled, sim_matrix = build_similarity(df)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Spotify Audio Recommendation System")
st.write(
    "Content-based recommendations built from audio features "
    "(danceability, energy, valence, tempo, and more) using cosine similarity."
)

tab_overview, tab_explore, tab_recommend = st.tabs(
    ["Data Overview", "Explore", "Recommendations"]
)

# ----------------------------------------------------------------------------
# Overview tab
# ----------------------------------------------------------------------------
with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total tracks", len(df))
    col2.metric("Liked", int((df["liked"] == 1).sum()))
    col3.metric("Disliked", int((df["liked"] == 0).sum()))

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Summary statistics")
    st.dataframe(df[FEATURES].describe().T, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Missing values")
        st.dataframe(df.isnull().sum().rename("null_count"))
    with c2:
        st.subheader("Duplicate rows")
        st.write(int(raw_df.duplicated().sum()))

# ----------------------------------------------------------------------------
# Explore tab
# ----------------------------------------------------------------------------
with tab_explore:
    st.subheader("Liked vs. disliked distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="liked", palette=[ACCENT, ACCENT_2], ax=ax)
    ax.set_xlabel("Liked (0 = No, 1 = Yes)")
    ax.set_ylabel("Number of tracks")
    st.pyplot(fig)
    plt.close(fig)

    feat_choice = st.selectbox(
        "Feature to inspect", FEATURES, index=FEATURES.index("danceability")
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Distribution of {feat_choice}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x=feat_choice, kde=True, color=ACCENT, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.subheader(f"{feat_choice} — liked vs. disliked")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(
            data=df, x="liked", y=feat_choice,
            palette=[ACCENT, ACCENT_2], ax=ax,
        )
        ax.set_xlabel("Liked (0 = No, 1 = Yes)")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Danceability vs. valence")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="danceability", y="valence", hue="liked",
        palette=[ACCENT_2, GOOD], ax=ax,
    )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation heatmap")
    fig, ax = plt.subplots(figsize=(11, 8))
    corr = df[FEATURES + ["liked"]].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------------
# Recommendations tab
# ----------------------------------------------------------------------------
with tab_recommend:
    st.subheader("1. Similar tracks to a chosen song")
    song_options = {track_label(df, i): i for i in df.index}
    chosen_label = st.selectbox("Choose a track", list(song_options.keys()))
    chosen_idx = song_options[chosen_label]
    n_song = st.slider("Number of recommendations", 3, 15, 5, key="song_n")

    song_recs = recommend_from_song(sim_matrix, df, chosen_idx, n_song)
    st.dataframe(song_recs, use_container_width=True)

    st.divider()

    st.subheader("2. Recommendations from your liked tracks")
    st.caption("Builds an average profile from every track marked `liked = 1` "
               "and ranks the remaining tracks by similarity to it.")
    n_profile = st.slider("Number of recommendations", 3, 20, 10, key="profile_n")

    profile_recs, profile_vec = recommend_from_profile(X_scaled, df, n_profile)
    if profile_recs.empty:
        st.warning("No tracks marked `liked = 1` in this dataset.")
    else:
        st.dataframe(profile_recs, use_container_width=True)

        st.subheader("Where the recommendations sit")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df, x="danceability", y="energy", hue="liked",
            palette=[ACCENT_2, "#B0B0B0"], alpha=0.6, ax=ax,
        )
        ax.scatter(
            profile_recs["danceability"], profile_recs["energy"],
            color=GOOD, s=140, marker="D", edgecolor="black",
            label="Recommended",
        )
        ax.legend()
        ax.set_title("Danceability vs. energy — recommendations highlighted")
        st.pyplot(fig)
        plt.close(fig)
