import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Spotify Music Style Analyzer",
    page_icon="🎵",
    layout="wide"
)

# ==========================================
# LOAD MODEL FILES
# ==========================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 🔥 FIX: correct feature order from training
FEATURES = list(scaler.feature_names_in_)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🎵 Spotify Analyzer")

st.sidebar.markdown("### About")
st.sidebar.write(
    """
    This application analyzes song characteristics
    and predicts the music style that best matches
    those audio features.
    """
)

st.sidebar.markdown("### Model Information")

st.sidebar.metric("Genres Learned", "114")
st.sidebar.metric("PCA Components", str(pca.n_components_))

st.sidebar.info(
    """
    PCA was used to reduce dimensions while
    preserving 95% of the original information.
    """
)

# ==========================================
# TITLE
# ==========================================

st.title("🎵 Spotify Music Style Analyzer")

st.markdown(
    """
    Describe a song using the controls below and discover
    which music style it most closely resembles.
    """
)

# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Analyze Song",
        "📊 Model Info",
        "📖 Feature Guide"
    ]
)

# ==========================================
# TAB 1
# ==========================================

with tab1:

    st.subheader("Song Characteristics")

    col1, col2 = st.columns(2)

    with col1:

        popularity = st.slider("🔥 Popularity", 0, 100, 50)

        duration_ms = st.number_input(
            "⏱ Duration (milliseconds)",
            min_value=10000,
            value=200000
        )

        explicit = st.selectbox("🚫 Explicit Content", [0, 1])

        danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("⚡ Energy", 0.0, 1.0, 0.5)
        key = st.slider("🎹 Musical Key", 0, 11, 5)
        loudness = st.slider("🔊 Loudness", -60.0, 5.0, -10.0)

    with col2:

        mode = st.selectbox("🎼 Mode", [0, 1])

        speechiness = st.slider("🎤 Speechiness", 0.0, 1.0, 0.1)
        acousticness = st.slider("🎸 Acousticness", 0.0, 1.0, 0.5)
        instrumentalness = st.slider("🎹 Instrumentalness", 0.0, 1.0, 0.0)
        liveness = st.slider("🎙 Liveness", 0.0, 1.0, 0.2)
        valence = st.slider("😊 Positivity", 0.0, 1.0, 0.5)
        tempo = st.slider("🥁 Tempo", 50.0, 250.0, 120.0)
        time_signature = st.slider("🎵 Time Signature", 1, 7, 4)

    st.markdown("")

    if st.button("🎵 Analyze Music Style", use_container_width=True):

        # =========================
        # FIX: duration_min handled safely
        # =========================
        duration_min = duration_ms / 60000

        input_df = pd.DataFrame(
            [[
                popularity,
                duration_ms,
                duration_min,
                explicit,
                danceability,
                energy,
                key,
                loudness,
                mode,
                speechiness,
                acousticness,
                instrumentalness,
                liveness,
                valence,
                tempo,
                time_signature
            ]],
            columns=[
                "popularity",
                "duration_ms",
                "duration_min",
                "explicit",
                "danceability",
                "energy",
                "key",
                "loudness",
                "mode",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "liveness",
                "valence",
                "tempo",
                "time_signature"
            ]
        )

        # =========================
        # FIX: column order match
        # =========================
        input_df = input_df.reindex(columns=FEATURES)

        scaled_data = scaler.transform(input_df)
        pca_data = pca.transform(scaled_data)

        prediction = model.predict(pca_data)

        music_style = label_encoder.inverse_transform(prediction)[0]

        st.success(
            f"🎵 This song most closely matches: {music_style}"
        )

        if hasattr(model, "predict_proba"):

            confidence = np.max(
                model.predict_proba(pca_data)
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2%}"
            )

        st.info(
            """
            The prediction is based on the song's
            audio characteristics such as energy,
            tempo, loudness, danceability,
            acousticness and other Spotify features.
            """
        )

# ==========================================
# TAB 2
# ==========================================

with tab2:

    st.subheader("Model Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Original Features", "15")
    col2.metric("PCA Components", str(pca.n_components_))
    col3.metric("Variance Retained", "95%")

    st.markdown("---")

    st.write(
        """
        The model was trained using Spotify audio
        characteristics from more than 113,000 songs.

        PCA (Principal Component Analysis) was used
        to reduce dimensions before training a
        Random Forest classifier.
        """
    )

# ==========================================
# TAB 3
# ==========================================

with tab3:

    st.subheader("Audio Feature Guide")

    with st.expander("💃 Danceability"):
        st.write("How suitable a track is for dancing.")

    with st.expander("⚡ Energy"):
        st.write("Represents intensity and activity.")

    with st.expander("🎤 Speechiness"):
        st.write("Measures the presence of spoken words.")

    with st.expander("🎸 Acousticness"):
        st.write("Likelihood that the song is acoustic.")

    with st.expander("🎹 Instrumentalness"):
        st.write("Likelihood that a track contains no vocals.")

    with st.expander("😊 Positivity"):
        st.write("How positive or happy a song sounds.")

    with st.expander("🥁 Tempo"):
        st.write("Speed of the track in beats per minute.")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Built with Streamlit • PCA • Random Forest • Spotify Audio Features"
)