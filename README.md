# NLP

import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score

import json

class Review:
    def __init__(self, category, text):
        self.category = category
        self.text = text    
        
class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
    
    def get_text(self):
        return [x.text for x in self.reviews]
    
    def get_y(self):
        return [x.category for x in self.reviews]


train_reviews = []
all_categories = []
for file in os.listdir('./NLP/data/training'):
    category = file.strip('train_').split('.')[0]
    all_categories.append(category)
    with open(f'./NLP/data/training/{file}') as f:
        for line in f:
            review_json = json.loads(line)
            review = Review(category, review_json['reviewText'])
            train_reviews.append(review)

train_container = ReviewContainer(train_reviews)




test_reviews = []
for file in os.listdir('./NLP/data/test'):
    category = file.strip('test_').split('.')[0]
    with open(f'./NLP/data/test/{file}') as f:
        for line in f:
            review_json = json.loads(line)
            review = Review(category, review_json['reviewText'])
            test_reviews.append(review)
            
test_container = ReviewContainer(test_reviews)

from sklearn import svm

corpus = train_container.get_text()
vectorizer = CountVectorizer(binary=True)
train_x = vectorizer.fit_transform(corpus) # training text converted to vector

clf = svm.SVC(kernel='linear')
clf.fit(train_x, train_container.get_y())

# make sure to convert test text to vector form
test_corpus = test_container.get_text()
test_x = vectorizer.transform(test_corpus)

print("Overall Accuracy:", clf.score(test_x, test_container.get_y()))

y_pred = clf.predict(test_x)

print("f1 scores by category")
print(all_categories)
print(f1_score(test_container.get_y(), y_pred, average=None, labels=all_categories))
