# 🗂️ Customer Complaint Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An NLP pipeline that automatically classifies consumer financial complaints into one of four categories using TF-IDF features and traditional machine learning models. Includes a fully interactive web app built with Streamlit.

> Built on the publicly available [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).

---

## 📌 What It Does

When a customer submits a complaint, this model reads the text and predicts which department it belongs to — no manual triage needed.

**Supported categories:**

| Category | Example complaint |
|---|---|
| Credit reporting | "My credit report shows a paid debt as still open." |
| Debt collection | "A collector keeps calling me about a debt that isn't mine." |
| Mortgage | "My servicer applied my payment to the wrong account." |
| Credit card | "I was charged a fee I never agreed to." |

---

## 📁 Project Structure

```
Customer-Classification/
│
├── Dataset/
│   └── customercomplaints.csv       
│
├── Models/
│   ├── logistic_regression.pkl
│   ├── linear_svm.pkl
│   ├── random_forest.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── Notebooks/        
│   └── Train.ipynb                 
│
├── app.py                           
├── predict.py                       
├── requirements.txt
├── run.bat                        
└── README.md
```

---

## ⚙️ Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/Customer-Classification.git
cd Customer-Classification
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download NLTK data** *(auto-runs on first use, or run manually)*
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## 🚀 Usage

### Option 1 — Web App (Streamlit)

```bash
streamlit run app.py
```

Then open your browser and go to:
```
http://localhost:8501
```

The web app lets you type any complaint and instantly see the predicted category with confidence scores for all four classes.

### Option 2 — Run Predictions (CLI)

```bash
python predict.py
```

```
==================================================
   Customer Complaint Classifier
==================================================

Enter complaint text (or 'quit' to exit): My credit report shows a
closed account that keeps appearing as open and I cannot get it fixed.

-> Predicted Category: Credit reporting
--------------------------------------------------
```

### Option 3 — Double Click (Windows)

Double click `run.bat` to launch the web app directly without opening a terminal manually.

### Retrain the Models

Open and run `Notebooks/Train.ipynb` top to bottom. Trained models are saved automatically to `Models/`.

---

## 🔬 How It Works

```
Raw text
   │
   ▼
Lowercase → Strip URLs / XXXX tokens / digits / punctuation
   │
   ▼
Tokenise (NLTK word_tokenize)
   │
   ▼
Remove stopwords → Lemmatise (WordNetLemmatizer)
   │
   ▼
TF-IDF Vectorisation (5,000 features, unigrams + bigrams)
   │
   ▼
Classifier → Predicted Category + Confidence Scores
```

---

## 📊 Results

Trained on 80% of ~26,000 filtered records, evaluated on the remaining 20%.

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | ~89% | Used for inference — fast and interpretable |
| Linear SVM | ~90% | Marginally best on sparse text features |
| Random Forest | ~85% | Included as baseline; weaker on high-dim sparse data |

The best model is selected automatically at the end of `Train.ipynb` and its full classification report and confusion matrix are displayed.

---

## 🖥️ Web App Features

- Clean and simple interface built with Streamlit
- Text input box for entering any complaint
- Instant category prediction on button click
- Confidence score progress bars for all 4 categories
- Sidebar with model info and category descriptions
- No setup required beyond installing dependencies

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **NLP:** NLTK (tokenisation, stopwords, lemmatisation)
- **Features:** scikit-learn TF-IDF Vectorizer
- **Models:** Logistic Regression, Linear SVC, Random Forest
- **Web App:** Streamlit
- **Serialisation:** joblib
- **Visualisation:** matplotlib, seaborn

---

## 🤝 Contributing

Contributions are welcome. To propose a change:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-idea`)
5. Open a Pull Request

For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Shoaib Amir**
- GitHub: [shoaibamir5](https://github.com/shoaibamir5)

---

> ⭐ If you found this useful, consider giving the repo a star!
