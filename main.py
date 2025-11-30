import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import re

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px; border-radius: 10px; }
    .stButton button { background-color: #ff4b4b; color: white; padding: 10px 24px; width: 100%; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# ---------- LOAD ASSETS ----------
@st.cache_resource
def load_assets():
    try:
        # compile=False prevents the Python 3.13 optimizer crash
        model = load_model('simple_rnn_imdb.keras', compile=False)
        word_index = imdb.get_word_index()
        return model, word_index
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, word_index = load_assets()

# ---------- PREPROCESSING (CRITICAL FIXES) ----------
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation but keep spaces (fixes "boring.")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    words = text.split()
    
    # 3. Start with token 1 (<START>) - CRITICAL FOR ACCURACY
    encoded_review = [1] 
    
    for word in words:
        if word in word_index:
            # Shift by 3 (0=PAD, 1=START, 2=UNK, 3=UNUSED)
            idx = word_index[word] + 3
            
            # Cap at 10,000 (Model input size limit)
            # If word rank is too rare, treat as Unknown (2)
            if idx >= 10000:
                encoded_review.append(2)
            else:
                encoded_review.append(idx)
        else:
            # Unknown word -> 2
            encoded_review.append(2)

    # 4. Pad sequence to length 500
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# ---------- UI & LOGIC ----------
st.title('🎬 Movie Review Sentiment')
st.markdown("Type a review to check if it's **Positive** or **Negative**.")

# Quick test buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("😊 Positive Example"):
        st.session_state.review_text = "The movie was absolutely fantastic! The acting was great and the plot was thrilling."
with col2:
    if st.button("😡 Negative Example"):
        st.session_state.review_text = "Really bad don't watch this. Complete waste of time and money."

if 'review_text' not in st.session_state:
    st.session_state.review_text = ""

user_input = st.text_area("Review Text:", value=st.session_state.review_text, height=150)

if st.button('Analyze Sentiment', type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    elif model is None:
        st.error("Model not found.")
    else:
        with st.spinner('Thinking...'):
            processed_data = preprocess_text(user_input)
            prediction = model.predict(processed_data)
            score = prediction[0][0]

            # Logic: > 0.5 is Positive, < 0.5 is Negative
            if score > 0.5:
                sentiment = "Positive"
                emoji = "👍"
                color = "#2ecc71" # Green
            else:
                sentiment = "Negative"
                emoji = "👎"
                color = "#ff4b4b" # Red

            st.markdown("---")
            
            # Result Columns
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; text-align: center;">
                    <h2 style="color: {color}; margin:0;">{emoji} {sentiment}</h2>
                    <p style="color: gray; margin-top: 5px;">Score: {score:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.write("Confidence Meter:")
                st.progress(float(score))
                st.caption("0 (Negative) ------------------------- 1 (Positive)")
