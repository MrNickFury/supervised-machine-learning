import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetimeZ

import numpy as np

uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
data = pd.read_csv(uri)
np.random.seed(8)

rename_values = {
	'yes' : 1,
	'no' : 0
}

data.sold = data.sold.map(rename_values)

today_year = datetime.today().year
data.model_year = today_year - data.model_year

x = data[['mileage_per_year', 'model_year', 'price']]
y = data[['sold']]

raw_train_x, raw_test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.25, stratify=y)

#uses deviance formulae to set this up
scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

dummy = DummyClassifier()
dummy.fit(train_x, train_y)
dummy_accuracy = dummy.score(test_x, test_y) * 100

print('DummyClassifier model Accuracy: %.2f' % dummy_accuracy + '%')

model = SVC()
model.fit(train_x, train_y)

predictions = model.predict(test_x)
accuracy = accuracy_score(test_y, predictions) * 100

print('LinearSVC model Accuracy: %.2f' % accuracy + '%')

