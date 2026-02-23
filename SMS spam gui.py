import tkinter as tk
from tkinter import messagebox
import joblib

model = joblib.load('sms_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

class SMSClassifier:
    def __init__(self, model, tfidf):
        self.model = model
        self.tfidf = tfidf
    
    def predict(self, texts):
        X_tfidf = self.tfidf.transform(texts)
        preds = self.model.predict(X_tfidf)
        return ["Spam" if p==1 else "Not Spam" for p in preds]

classifier = SMSClassifier(model, tfidf)

def predict_spam():
    text = entry.get()
    if not text.strip():
        messagebox.showwarning("Warning", "Please enter a message")
        return
    prediction = classifier.predict([text])[0]
    result_label.config(text=f"Prediction: {prediction}")

root = tk.Tk()
root.title("SMS Spam Classifier")
root.geometry("500x250")
title_label = tk.Label(root, text="SMS Spam Classifier", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

input_label = tk.Label(root, text="Enter your SMS message:")
input_label.pack()

entry = tk.Entry(root, width=60)
entry.pack(pady=5)

predict_button = tk.Button(root, text="Predict", command=predict_spam, bg="blue", fg="white")
predict_button.pack(pady=10)
result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14))
result_label.pack(pady=10)

root.mainloop()