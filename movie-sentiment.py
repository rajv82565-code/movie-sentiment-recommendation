##load dataset 
import pandas as pd

df = pd.read_csv("IMDB Dataset.csv/IMDB Dataset.csv")
df = df.sample(2000)  # Use a sample for faster training

##Clean Text
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean(text):
    text = re.sub(r'<.*?>','',text)
    text = re.sub(r'[^a-zA-Z]',' ',text)
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

df['clean_review'] = df['review'].apply(clean)

##TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_review'])
y = df['sentiment'].map({'positive':1,'negative':0})

##Train Logistic Regression
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(X,y)

##Train Naive Bayes
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X,y)

##Sentiment Prediction Function

def predict_sentiment(review_text):

    review_text = clean(review_text)
    vec = tfidf.transform([review_text])

    log_pred = log_model.predict(vec)[0]
    nb_pred = nb_model.predict(vec)[0]

    # Voting logic
    final = round((log_pred + nb_pred) / 2)

    if final == 1:
        return "Positive"
    else:
        return "Negative"

if __name__ == "__main__":
    print("--- Movie Sentiment Analysis ---")
    print("Training models... (using 2000 samples)")
    
    # Simple test cases
    test_reviews = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Worst experience ever. The plot was boring and the acting was terrible.",
        "It was okay, but I've seen better movies."
    ]
    
    for review in test_reviews:
        sentiment = predict_sentiment(review)
        print(f"\nReview: {review}")
        print(f"Predicted Sentiment: {sentiment}")