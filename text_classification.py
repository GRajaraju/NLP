


'''A simple program to classify text.
  The implementation is based on my understanding from the below paper:
  https://arxiv.org/pdf/1410.5329v3.pdf
'''
# In this program we will perform the sentimental analysis using Naive Bayes.

import numpy as np
import pandas as pd
import re
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("customers_reviews.tsv",delimiter='\t',quoting = 3)

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

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

# to classify new reviews, pass on the new text to the below function where we are basically preprocessing the text suitable
# to the model. these steps are same we performed above.

def text_preprocess(new_review):
  new_review = re.sub('[^a-zA-Z]',' ',new_review)
  new_review = new_review.lower()
  new_review = new_review.split()
  new_review = [ps.stem(word) for word in new_review if not word in set(stopwords.words('english'))]
  new_review = " ".join(new_review)
  corpus.append(new_text_review)
  new_review = cv.fit_transform(corpus).toarray()
  t1 = new_review[1001] # specify the newly added sample (should be the last row in the corpus)
  return classifier.predict([[t1]])

  
  
