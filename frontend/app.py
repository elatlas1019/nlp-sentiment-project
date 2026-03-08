import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Sentiment Analysis", page_icon="🔮", layout="centered")

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🔮 AI Sentiment Analysis")
    st.write("Enter a sentence below to analyze its sentiment using our AI model.")

    text_input = st.text_area("Your Text", placeholder="Type something like 'I love this project!' or 'I'm feeling a bit sad today.'")

    API_KEY = os.getenv("API_KEY")
    BACKEND_URL = "http://localhost:8000/predict"

    if st.button("Analyze Sentiment"):
        if text_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    headers = {"x-api-key": API_KEY}
                    response = requests.post(BACKEND_URL, json={"text": text_input}, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        pos = result.get("positive", 0)
                        neu = result.get("neutral", 0)
                        neg = result.get("negative", 0)
                        
                        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                        st.subheader("Result:")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Positive", f"{pos*100:.1f}%")
                        with col2:
                            st.metric("Neutral", f"{neu*100:.1f}%")
                        with col3:
                            st.metric("Negative", f"{neg*100:.1f}%")
                        
                        # Determine the dominant sentiment
                        sentiment_data = [
                            ("Positive", pos, st.success),
                            ("Neutral", neu, st.warning),
                            ("Negative", neg, st.error)
                        ]
                        winner_name, winner_score, winner_func = max(sentiment_data, key=lambda x: x[1])
                        
                        winner_func(f"Overall Sentiment: **{winner_name}** (Confidence: {winner_score*100:.1f}%)")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    st.info("Note: Make sure the backend server is running on http://localhost:8000")

if __name__ == "__main__":
    main()
