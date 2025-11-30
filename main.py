import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- LOAD ASSETS (CACHED) ----------
# We use cache_resource so we don't reload the model/data on every click
@st.cache_resource
def load_assets():
    try:
        # Load Model
        model = load_model('simple_rnn_imdb.keras', compile=False)
        
        # Load Word Index
        word_index = imdb.get_word_index()
        return model, word_index
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

model, word_index = load_assets()

# ---------- PREPROCESSING FUNCTION ----------
def preprocess_text(text):
    words = text.lower().split()
    # IMDB mapping logic: 
    # Words start at index 1. Index 0 is padding.
    # Indices 1, 2, 3 are reserved usually for <START>, <OOV>, <UNUSED>
    # Your logic used +3, which implies standard IMDB offset
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# ---------- MAIN INTERFACE ----------
st.title('🎬 Movie Review Sentiment')
st.markdown("Type a review below to see if the AI thinks it's **Positive** or **Negative**.")

# Sample buttons to quickly fill text
col1, col2 = st.columns(2)
with col1:
    if st.button("📝 Try Positive Example"):
        st.session_state.review_text = "The movie was absolutely fantastic! The acting was great and the plot was thrilling."
with col2:
    if st.button("📝 Try Negative Example"):
        st.session_state.review_text = "Terrible movie. Complete waste of time and money. The script was boring."

# Text Area (uses session state to update from buttons)
if 'review_text' not in st.session_state:
    st.session_state.review_text = ""

user_input = st.text_area(
    'Write your review here:', 
    value=st.session_state.review_text,
    height=150,
    placeholder="e.g. I really enjoyed this film..."
)

# ---------- PREDICTION LOGIC ----------
if st.button('Analyze Sentiment', type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    elif model is None:
        st.error("Model not loaded. Please check your files.")
    else:
        with st.spinner('Analyzing...'):
            # Preprocess
            preprocessed_input = preprocess_text(user_input)
            
            # Predict
            prediction = model.predict(preprocessed_input)
            score = prediction[0][0] # Value between 0 and 1
            
            # Determine Sentiment
            if score > 0.5:
                sentiment = "Positive"
                emoji = "👍"
                color = "green"
            else:
                sentiment = "Negative"
                emoji = "👎"
                color = "red"
            
            # Display Results
            st.markdown("---")
            st.subheader("Analysis Result:")
            
            # Layout for metrics
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown(f"<h2 style='color: {color};'>{emoji} {sentiment}</h2>", unsafe_allow_html=True)
                st.caption(f"Confidence Score: {score:.4f}")
            
            with res_col2:
                st.write("Sentiment Probability:")
                # Progress bar: 0 = Negative (Red), 1 = Positive (Green)
                st.progress(float(score))
                st.caption("0 (Negative) .............................. 1 (Positive)")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with TensorFlow & Streamlit</p>", unsafe_allow_html=True)

