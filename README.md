# SMS Spam Classifier

A simple **SMS Spam Detection** project using **Python**, **TF-IDF**, and **Naive Bayes** with a **GUI** for real-time predictions. The project can classify any SMS message as **Spam** or **Not Spam** and handles **imbalanced datasets**.

---

## Features

* Handles **class imbalance** using oversampling
* Uses **TF-IDF** with unigrams and bigrams for feature extraction
* Uses **Multinomial Naive Bayes** for classification
* Standalone **Tkinter GUI** to input SMS messages and get predictions
* Model and vectorizer saved for future use → **no CSV needed after initial training**

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/SMS-spam-classifier.git
cd SMS-spam-classifier
```

2. Install required packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

3. Make sure you have **Python 3.8+** installed.

---

## Technologies Used

* Python 3.x
* scikit-learn (Naive Bayes, TF-IDF)
* pandas, numpy (data handling)
* Tkinter (GUI)
* joblib (saving/loading model)

---

## Notes

* Minimal text preprocessing is applied to preserve important spam keywords (numbers, links, phrases).
* The classifier handles **imbalanced datasets** with oversampling of the spam class.
* For large datasets, you can replace Tkinter with **Streamlit** or a web app for better usability.

---

Do you want me to do that?
