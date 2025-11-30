import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import re

# ---------- PAGE CONFIGURATION ----------
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
        width: 100%;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- LOAD ASSETS (CACHED) ----------
@st.cache_resource
def load_assets():
    try:
        # Fix for Attribute Error: compile=False prevents optimizer crashes
        model = load_model('simple_rnn_imdb.keras', compile=False)
        
        # Load Word Index
        word_index = imdb.get_word_index()
        return model, word_index
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

model, word_index = load_assets()

# ---------- PREPROCESSING FUNCTION (FIXED) ----------
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation (fixes "boring." issue)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    words = text.split()
    
    encoded_review = []
    for word in words:
        if word in word_index:
            # IMDB indices are offset by 3 (0=pad, 1=start, 2=unknown, 3=unused)
            # 'the' is index 1 in word_index, but 4 in the model
            idx = word_index[word] + 3
            
            # Use specific index for Unknown (2) if the word index is too high
            # (Assuming model trained on top 10,000 words)
            if idx >= 10000:
                encoded_review.append(2)
            else:
                encoded_review.append(idx)
        else:
            # If word is completely new, mark as Unknown (2)
            encoded_review.append(2)

    # Pad the sequence
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# ---------- MAIN INTERFACE ----------
st.title('🎬 Movie Review Sentiment')
st.markdown("Type a review below to see if the AI thinks it's **Positive** or **Negative**.")

# Sample buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("📝 Try Positive Example"):
        st.session_state.review_text = "The movie was absolutely fantastic! The acting was great and the plot was thrilling."
with col2:
    if st.button("📝 Try Negative Example"):
        st.session_state.review_text = "Terrible movie. Complete waste of time and money. The script was boring."

# Text Area
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
        st.error("Model failed to load.")
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
                color = "#2ecc71" # Green
            else:
                sentiment = "Negative"
                emoji = "👎"
                color = "#ff4b4b" # Red
            
            # Display Results
            st.markdown("---")
            st.subheader("Analysis Result:")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown(f"""
                <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; text-align: center;">
                    <h2 style='margin:0; color: {color};'>{emoji} {sentiment}</h2>
                    <p style='margin:5px 0 0 0; color: grey;'>Score: {score:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with res_col2:
                st.write("Sentiment Probability:")
                st.progress(float(score))
                st.caption("0 (Negative) .............................. 1 (Positive)")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with TensorFlow & Streamlit</p>", unsafe_allow_html=True)
