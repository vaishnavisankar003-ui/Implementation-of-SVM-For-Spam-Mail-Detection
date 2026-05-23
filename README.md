# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset using pandas.

2.Convert text messages into numbers using TF-IDF Vectorizer.

3.Split the data into training and testing sets.

4.Train the Support Vector Machine model.

5.Evaluate the result using a Confusion Matrix.



## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: vaishnavi s
RegisterNumber:  212225230289
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1','v2']]
data.columns = ['label','message']

# Convert labels
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Features and target
X = data['message']
y = data['label']

# TF-IDF conversion
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM Spam Detection")
plt.show()
```

## Output:

<img width="506" height="407" alt="image" src="https://github.com/user-attachments/assets/31a8b358-2bce-4748-960c-22f37cf718d1" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
