
# UCI Machine Learning repository https://archive.ics.uci.edu/ml/machine-learning-databases/00228/
import pandas as pd
# from io import BytesIO
# from zipfile import ZipFile
# from urllib.request import urlopen
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score


# fetch from the zip from the url
#url = urlopen(
#   "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip")
#zipfile = ZipFile(BytesIO(url.read()))
f = open("./SMSSpamCollection")
df = pd.read_table(f,
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])

# Print out first 5 columns
df.head()

# map ham and spam values to binary
df['label'] = df.label.map({'ham': 0, 'spam': 1})
# check sizes of dataframe
df.shape


'''
Split the data into training and test sets
'''
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
print('Number of rows in the output train set: {}'.format(y_train.shape[0]))



'''
The code for this segment is in 2 parts. Firstly, we are learning a vocabulary dictionary for the training data
and then transforming the data into a document-term matrix; secondly, for the testing data we are only
transforming the data into a document-term matrix.
We will provide the transformed data to students in the variables 'training_data' and 'testing_data'.
'''

# Instantiate the CountVectorizer method

count_vector = CountVectorizer()
# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)
# Transform testing data and return the matrix.
# Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)


'''
naive bayes
'''
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)


predictions = naive_bayes.predict(testing_data)
print(predictions)


'''
Check performance
'''
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
