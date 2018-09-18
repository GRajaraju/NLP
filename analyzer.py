from flask import Flask, render_template, request
# import numpy as np
# import pandas as pd
# import re
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.cross_validation import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import confusion_matrix

app = Flask(__name__)

def preprocess_training():
    dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting = 3)

    corpus = []

    for i in range(0,1000):
        review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = " ".join(review)
        corpus.append(review)

    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()

    y = dataset.iloc[:,1].values

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

    classifier = GaussianNB()
    classifier.fit(X_train,y_train)

    return ps, corpus, cv, classifier


def text_preprocess(new_review):
    ps, corpus, cv, classifier = preprocess_training()
    new_review = re.sub('[^a-zA-Z]',' ',new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    new_review = [ps.stem(word) for word in new_review if not word in set(stopwords.words('english'))]
    new_review = " ".join(new_review)
    corpus.append(new_review)
    new_review = cv.fit_transform(corpus).toarray()
    t1 = new_review[1000] # specify the newly added sample (should be the last row in the corpus)
    return classifier.predict([t1])


@app.route("/")
def index():
    return render_template("home.html")

# @app.route("/analyse",methods=['POST'])
# def analyse():
#     comments = request.form['message']
#     new_review = int(text_preprocess(comments))
#     if new_review == 1:
#         return "Positive feedback."
#     else:
#         return "Negative feedback."


if __name__ == "__main__":
    app.run(debug=True)
