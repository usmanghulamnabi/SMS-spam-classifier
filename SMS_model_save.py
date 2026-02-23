import numpy as np
import pandas as pd
import re
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample


df = pd.read_csv('spam.csv', encoding='latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.columns = ['label', 'message']
df['label'] = LabelEncoder().fit_transform(df['label'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text
df['message'] = df['message'].apply(clean_text)

df_ham = df[df.label == 0]
df_spam = df[df.label == 1]
df_spam_upsampled = resample(df_spam, replace=True, n_samples=len(df_ham), random_state=42)
df_balanced = pd.concat([df_ham, df_spam_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
X = df_balanced['message']
y = df_balanced['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

joblib.dump(model, 'sms_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Model and vectorizer saved successfully!")