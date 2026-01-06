import streamlit as st
import joblib
import re
import nltk
import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -----------------------------
# NLTK
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------
# LOAD MODEL & VECTORIZER
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "model", "sentiment_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "model", "tfidf.pkl"))

stop_words = set(stopwords.words('english'))

# -----------------------------
# PREPROCESS FUNCTION
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Review Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# GLOBAL BACKGROUND + UI STYLING
# -----------------------------
st.markdown("""
<style>
/* FULL PAGE BACKGROUND */
html, body, [data-testid="stApp"] {
    background-color: #F0F7FF;
}

/* HEADER */
.header-box {
    background: linear-gradient(90deg, #5B86E5, #36D1DC);
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    color: white;
}

/* CARD STYLE */
.card {
    background-color: #FFFFFF;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.08);
    margin-bottom: 22px;
}

/* BUTTONS */
.stButton>button {
    background-color: #5B86E5;
    color: white;
    border-radius: 10px;
    padding: 8px 20px;
    border: none;
    font-weight: 500;
}
.stButton>button:hover {
    background-color: #3f6ad8;
}

/* FOOTER */
.footer {
    text-align: center;
    font-size: 12px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="header-box">
    <h1>Customer Review Sentiment Analyzer</h1>
    <p>Analyze customer feedback using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# EXAMPLE REVIEWS
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("✨ Try Example Reviews")

col1, col2, col3 = st.columns(3)

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

with col1:
    if st.button("😊 Positive"):
        st.session_state.review_text = "I absolutely loved the product. It works perfectly!"

with col2:
    if st.button("😐 Neutral"):
        st.session_state.review_text = "The product is okay and does the job."

with col3:
    if st.button("😞 Negative"):
        st.session_state.review_text = "Very disappointing experience. Not worth the money."

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# INPUT SECTION
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("✍️ Enter Your Review")
review = st.text_area("", value=st.session_state.review_text, height=130)
analyze = st.button("🔍 Analyze Sentiment")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# ANALYSIS SECTION
# -----------------------------
if analyze:
    if review.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        processed = preprocess(review)
        vector = tfidf.transform([processed])

        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        labels = model.classes_
        confidence = max(probabilities) * 100

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Prediction Result")

        if str(prediction).lower() == "positive":
            st.success(f"😊 Sentiment: {prediction}")
        elif str(prediction).lower() == "negative":
            st.error(f"😞 Sentiment: {prediction}")
        else:
            st.info(f"😐 Sentiment: {prediction}")

        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))

        # Probability Chart
        fig, ax = plt.subplots()
        ax.bar([str(l) for l in labels], probabilities * 100)
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # CSV Download
        prob_dict = {str(l): round(p * 100, 2) for l, p in zip(labels, probabilities)}

        result_df = pd.DataFrame({
            "Review": [review],
            "Predicted Sentiment": [prediction],
            "Confidence (%)": [round(confidence, 2)],
            **{f"{k} (%)": [v] for k, v in prob_dict.items()}
        })

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Result CSV",
            csv,
            "sentiment_prediction.csv",
            "text/csv"
        )

        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
    <hr>
    Mini Project | Customer Review Sentiment Analysis<br>
    Developed by <b>Chithra Kurma</b>
</div>
""", unsafe_allow_html=True)