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
nltk.download('punkt_tab', quiet=True)

lr    = joblib.load("Models/logistic_regression.pkl")
tfidf = joblib.load("Models/tfidf_vectorizer.pkl")
le    = joblib.load("Models/label_encoder.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # strip URLs
    text = re.sub(r'\bxx+\b', '', text)           # remove XXXX placeholders common in this dataset
    text = re.sub(r'\d+', '', text)               # drop all digits
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


def predict_complaint(raw_text):
    cleaned   = clean_text(raw_text)
    processed = preprocess(cleaned)
    vec       = tfidf.transform([processed])
    pred      = lr.predict(vec)[0]
    return le.inverse_transform([pred])[0]


print("=" * 50)
print("   Customer Complaint Classifier")
print("=" * 50)

while True:
    print()
    user_input = input("Enter complaint text (or 'quit' to exit): ").strip()

    if user_input.lower() == 'quit':
        print("Exiting. Bye!")
        break

    if not user_input:
        print("Nothing entered, try again.")
        continue

    category = predict_complaint(user_input)
    print(f"-> Predicted Category: {category}")
    print("-" * 50)
