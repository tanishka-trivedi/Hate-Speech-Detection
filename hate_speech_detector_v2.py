import pandas as pd
import numpy as np
import re
import string
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words("english"))

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("twitter_data.csv")
df['labels'] = df['class'].map({
    0: "Hate Speech",
    1: "Offensive Language",
    2: "No Hate or Offensive Speech"
})
df = df[['tweet', 'labels']]

# ── 2. Clean ──────────────────────────────────────────────────────────────────
def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # FIX 1: Do NOT remove stopwords for short/positive sentences
    # Keeping stopwords gives the model more context
    # text = [word for word in text.split() if word not in stopword]
    text = text.split()
    text = [stemmer.stem(word) for word in text]
    return " ".join(text)

df['tweet'] = df['tweet'].apply(clean)

# ── 3. Vectorize ──────────────────────────────────────────────────────────────
X_text = df['tweet']
y = df['labels']

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))  # FIX 2: bigrams
X = tfidf.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# ── 4. FIX 3: Use class_weight to handle imbalance ───────────────────────────
# Option A: Naive Bayes with balanced priors (manually set)
class_counts = y_train.value_counts()
total = len(y_train)
# Invert class frequency to balance priors
balanced_priors = {
    cls: total / (len(class_counts) * count)
    for cls, count in class_counts.items()
}
prior_values = np.array([balanced_priors[c] for c in sorted(balanced_priors)])
prior_values = prior_values / prior_values.sum()  # normalize

nb_model = MultinomialNB(class_prior=prior_values)
nb_model.fit(X_train, y_train)

# Option B: Logistic Regression with class_weight='balanced' (recommended)
lr_model = LogisticRegression(
    class_weight='balanced',   # automatically handles imbalance
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train, y_train)

# ── 5. Evaluate both models ───────────────────────────────────────────────────
for name, model in [("Naive Bayes (balanced priors)", nb_model),
                    ("Logistic Regression (balanced)", lr_model)]:
    y_pred = model.predict(X_test)
    print(f"\n── {name} ──────────────────────────")
    print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))

# ── 6. Predict function (uses best model: Logistic Regression) ────────────────
def predict(text: str, model=lr_model) -> str:
    cleaned = clean(text)
    vector = tfidf.transform([cleaned])
    return model.predict(vector)[0]

# ── 7. Demo ───────────────────────────────────────────────────────────────────
samples = [
    "you are awesome",
    "you are nice",
    "I hate you so much",
    "This is offensive garbage",
    "Have a great day!",
]

print("\n── Demo Predictions (Logistic Regression) ──────")
for s in samples:
    print(f"  '{s}'  →  {predict(s)}")
