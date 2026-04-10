import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)


st.set_page_config(
    page_title="Complaint Classifier",
    page_icon="📋",
    layout="centered"
)

@st.cache_resource
def load_models():
    lr    = joblib.load("Models/logistic_regression.pkl")
    tfidf = joblib.load("Models/tfidf_vectorizer.pkl")
    le    = joblib.load("Models/label_encoder.pkl")
    return lr, tfidf, le

lr, tfidf, le = load_models()

def predict_complaint(text):
    # Defined inside to avoid NLTK loading conflict
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\bxx+\b', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    final  = ' '.join(tokens)
    vec    = tfidf.transform([final])
    pred   = lr.predict(vec)[0]
    label  = le.inverse_transform([pred])[0]
    return label

category_info = {
    'Credit reporting' : {'icon': '📊', 'color': '#1f77b4'},
    'Debt collection'  : {'icon': '💳', 'color': '#d62728'},
    'Mortgage'         : {'icon': '🏠', 'color': '#2ca02c'},
    'Credit card'      : {'icon': '💰', 'color': '#ff7f0e'},
}

st.title("📋 Customer Complaint Classifier")
st.markdown("Enter a customer complaint below and the model will classify it into the correct category.")
st.markdown("---")

complaint_text = st.text_area(
    "Enter Complaint Text",
    height=150,
    placeholder="e.g. My mortgage payment was not processed and the bank is not responding..."
)

if st.button("Classify Complaint", type="primary"):
    if complaint_text.strip() == '':
        st.warning("Please enter a complaint first.")
    else:
        with st.spinner("Classifying..."):
            result = predict_complaint(complaint_text)
            info   = category_info.get(result, {'icon': '📌', 'color': '#888888'})

        st.markdown("---")
        st.subheader("Result")

        st.markdown(
            f"""
            <div style="
                background-color: {info['color']}22;
                border-left: 5px solid {info['color']};
                padding: 20px;
                border-radius: 8px;
                margin-top: 10px;">
                <h2 style="color: {info['color']}; margin:0;">
                    {info['icon']} {result}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("Confidence Scores")
        probs = lr.predict_proba(tfidf.transform([complaint_text]))[0]
        for i, cls in enumerate(le.classes_):
            info_cls = category_info.get(cls, {'icon': '📌', 'color': '#888888'})
            st.markdown(f"{info_cls['icon']} **{cls}**")
            st.progress(float(probs[i]))
            st.markdown(f"`{probs[i]*100:.1f}%`")

st.sidebar.title("About")
st.sidebar.info(
    "This app classifies customer complaints into one of 4 categories using "
    "a Machine Learning model trained on the CFPB Consumer Complaint Dataset."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Categories")
for cat, info in category_info.items():
    st.sidebar.markdown(f"{info['icon']} {cat}")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
st.sidebar.markdown("- **Algorithm:** Logistic Regression")
st.sidebar.markdown("- **Features:** TF-IDF (5000 features)")
st.sidebar.markdown("- **Ngrams:** Unigrams + Bigrams")
st.sidebar.markdown("- **Dataset:** CFPB Consumer Complaints")
st.sidebar.markdown("- **Training Size:** 30,000 samples")