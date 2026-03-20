# Hate-Speech-Detection

# 🚫 Hate Speech Detection — NLP Classifier

A machine learning project that classifies tweets into three categories: **Hate Speech**, **Offensive Language**, and **No Hate or Offensive Speech** using NLP preprocessing and supervised learning.

---

## 📌 Problem Statement

Social media platforms are flooded with harmful content. This project builds a text classifier to automatically detect hate speech and offensive language in tweets, helping moderate content at scale.

---

## 📂 Project Structure

```
hate-speech-detection/
│
├── twitter_data.csv              # Dataset (tweets with class labels)
├── hate_speech_detector.py       # v1 — Naive Bayes baseline model
├── hate_speech_detector_v2.py    # v2 — Improved model with class balancing
├── confusion_matrix.png          # Evaluation output (generated on run)
└── README.md
```

---

## 🗂️ Dataset

The dataset (`twitter_data.csv`) contains labelled tweets with the following class mapping:

| Class | Label |
|-------|-------|
| 0 | Hate Speech |
| 1 | Offensive Language |
| 2 | No Hate or Offensive Speech |

> **Note:** The dataset is heavily imbalanced — ~77% of tweets are labelled as Offensive Language, which is a key challenge addressed in v2.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### Install dependencies

```bash
pip install pandas numpy scikit-learn nltk matplotlib
```

```python
# Download NLTK stopwords (runs once)
import nltk
nltk.download('stopwords')
```

---

## 🔄 How It Works

### 1. Text Preprocessing (`clean()`)

Each tweet is cleaned through the following pipeline:

- Lowercasing
- Removing URLs, HTML tags, text inside brackets
- Removing punctuation and numbers
- Stemming using `SnowballStemmer`

> In **v2**, stopwords are intentionally **kept** to preserve sentence context (e.g., "you are nice" loses meaning without "you" and "are").

### 2. Feature Extraction

- **TF-IDF Vectorizer** with bigrams (`ngram_range=(1,2)`) — captures word pairs like "you are" and "are nice" for richer features.

### 3. Models

| Version | Model | Notes |
|---------|-------|-------|
| v1 | Multinomial Naive Bayes | Baseline, biased toward majority class |
| v2 | Naive Bayes (balanced priors) | Manually corrected class priors |
| v2 | **Logistic Regression** ✅ | `class_weight='balanced'` — recommended |

---

## 🚀 Usage

### Run the improved model (v2)

```bash
python hate_speech_detector_v2.py
```

### Predict on custom text

```python
from hate_speech_detector_v2 import predict

print(predict("you are awesome"))   # → No Hate or Offensive Speech
print(predict("I hate you"))        # → Hate Speech
```

---

## 📊 Evaluation

The model is evaluated using:

- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1 per class)
- **Confusion Matrix** (saved as `confusion_matrix.png`)

Sample output:

```
── Logistic Regression (balanced) ──────────────────────────
Accuracy : 89.XX%

              precision    recall  f1-score   support
  Hate Speech       0.xx      0.xx      0.xx       xxx
  Offensive Language  0.xx  0.xx      0.xx       xxx
  No Hate or Offensive Speech  0.xx  0.xx  0.xx  xxx
```

---

## 🐛 Known Issues & Fixes

| Issue | Cause | Fix Applied |
|-------|-------|-------------|
| "you are nice" predicted as Offensive | Class imbalance (77% offensive data) | `class_weight='balanced'` in Logistic Regression |
| Short sentences misclassified | Stopword removal strips all context | Stopwords retained in v2 |
| Weak features from single words | Unigrams only | Bigrams added via `ngram_range=(1,2)` |

---

## 🔧 Future Improvements

- [ ] Try transformer-based models (e.g., BERT, RoBERTa) for better context understanding
- [ ] Oversample minority classes using SMOTE
- [ ] Deploy as a REST API using FastAPI or Flask
- [ ] Add a simple web UI for live predictions
- [ ] Cross-validation for more robust evaluation

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Pandas](https://img.shields.io/badge/Pandas-Data-lightblue?logo=pandas)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Dataset sourced from the [Hate Speech and Offensive Language dataset](https://github.com/t-davidson/hate-speech-and-offensive-language) by T. Davidson et al.
- Built with [scikit-learn](https://scikit-learn.org/) and [NLTK](https://www.nltk.org/)
