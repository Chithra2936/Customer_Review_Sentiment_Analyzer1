import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/review.csv")
print(df.head())
print("Columns:", df.columns)

# Correct text and label columns
text_column = df.columns[3]   # review text
label_column = df.columns[2]  # sentiment
print("Text column:", text_column)
print("Label column:", label_column)

# Check class balance
print(df[label_column].value_counts())

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['clean_review'] = df[text_column].apply(clean_text)

# Tokenization & stopwords
stop_words = set(stopwords.words('english'))

def preprocess(text):
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['processed_review'] = df['clean_review'].apply(preprocess)

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['processed_review'])
y = df[label_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")
print("Model and TF-IDF Vectorizer saved successfully")
