from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd

SEED = 20  # commom number so that the randomness stay disabled when setting which lines from x and y to choose

model = LinearSVC()
uri = 'https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'
# returns a two-dimensional data structure with labeled axes.
data = pd.read_csv(uri)

x = data[['home', 'how_it_works', 'contact']]
y = data[['bought']]

# returns train and test valus based on x and y values given a certain proportion for test; stratify guaratees that
# the proportion of 0s and 1s for y values are the same in training and testing
train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.25, random_state=SEED, stratify=y)

model.fit(train_x, train_y)

predictions = model.predict(test_x)
accuracy = accuracy_score(test_y, predictions) * 100

print('LinearSVC model Accuracy: %.2f' % accuracy + '%')
