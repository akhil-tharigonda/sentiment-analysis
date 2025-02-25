import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK stopwords
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv(r'C:\Users\Windows\Downloads\Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Preprocessing the text data
corpus = []
ps = PorterStemmer()
for i in range(len(data)):
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=data['Review'][i])
    review = review.lower()
    review_words = review.split()
    review_words = [ps.stem(word) for word in review_words if word not in set(stopwords.words('english'))]
    review = ' '.join(review_words)
    corpus.append(review)

# Converting text to numerical data using CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Making predictions
y_pred = classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("---------SCORES--------")
print(f"Accuracy score: {round(accuracy * 100, 3)}%")
print(f"Precision score: {round(precision * 100, 3)}%")
print(f"Recall score: {round(recall * 100, 3)}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix')
plt.show()

# Tuning the alpha parameter for the Naive Bayes classifier
best_accuracy = 0.0
best_alpha = 0.0
for alpha in np.arange(0.1, 1.1, 0.1):
    temp_classifier = MultinomialNB(alpha=alpha)
    temp_classifier.fit(X_train, y_train)
    temp_y_pred = temp_classifier.predict(X_test)
    score = accuracy_score(y_test, temp_y_pred)
    print(f"Accuracy score for alpha={round(alpha, 1)}: {round(score * 100, 3)}%")
    if score > best_accuracy:
        best_accuracy = score
        best_alpha = alpha

print('----------------------------------------------------')
print(f"Best accuracy score: {round(best_accuracy * 100, 2)}% with alpha={round(best_alpha, 1)}")

# Final classifier with the best alpha
classifier = MultinomialNB(alpha=best_alpha)
classifier.fit(X_train, y_train)

# Function for predicting sentiment
def predict_sentiment(review):
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=review)
    review = review.lower()
    review_words = review.split()
    review_words = [ps.stem(word) for word in review_words if word not in set(stopwords.words('english'))]
    review = ' '.join(review_words)
    vectorized_review = cv.transform([review]).toarray()
    prediction = classifier.predict(vectorized_review)
    return 'Positive' if prediction == 1 else 'Negative'

# Testing the sentiment prediction function
sample_reviews = [
    'The food is really bad.',
    'Food was pretty bad and the service was very slow.',
    'The food was absolutely wonderful, from preparation to presentation, very pleasing.',
    'Food was average.'
]

for sample in sample_reviews:
    print(f"Review: {sample}\nSentiment: {predict_sentiment(sample)}\n")
