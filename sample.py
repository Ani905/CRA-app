from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
import streamlit as st

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def authenticate(given, predicted):
    if abs(given - predicted) <= 2:
        return (
            "<h3 style='color: green; text-align: center;'>✅ Genuine Review</h3>"
            "<p style='color: gray; text-align: center;'>Your rating aligns well with your review sentiment.</p>"
        )
    else:
        return (
            "<h3 style='color: red; text-align: center;'>⚠️ Ambiguous Review</h3>"
            "<p style='color: gray; text-align: center;'>There seems to be a mismatch between your rating and review sentiment.</p>"
        )

# Load model and tokenizer


# Initialize session state for rating and review
if "given" not in st.session_state:
    st.session_state.given = None
if "review" not in st.session_state:
    st.session_state.review = None

# Streamlit App Title
st.title("Product Feedback Authenticator")

# Star rating input
st.header("Please rate the product")
selected_rating = st.slider("Rate the product (1-5 stars):", min_value=1, max_value=5, step=1)
if st.button("Submit Star Rating", key="star_rating_button"):
    st.session_state.given = selected_rating
    st.success(f"Thank you for your {selected_rating}-star review!")

# Text box for review input
st.header("We would love to know your thoughts on the product")
review_text = st.text_area("Write your review here:")
if st.button("Submit Text Review", key="text_review_button"):
    if review_text.strip():
        st.session_state.review = review_text
        st.success("Your review has been submitted!")
    else:
        st.error("Please write a review before submitting.")

# Authenticate button
if st.button("Authenticate"):
    if st.session_state.given is None or st.session_state.review is None:
        st.error("Please submit both a star rating and a review before authenticating!")
    else:
        tokens = tokenizer.encode(st.session_state.review, return_tensors="pt")
        result = model(tokens)
        predicted = int(torch.argmax(result.logits)) + 1
        st.markdown(authenticate(st.session_state.given, predicted), unsafe_allow_html=True)
