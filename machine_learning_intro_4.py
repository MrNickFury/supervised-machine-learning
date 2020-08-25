from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

SEED = 88
np.random.seed(SEED) 
#train_test_split and SVC uses this as its default seed, so i only set 
#it and both of them are in the same page

model = SVC()
uri='https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'
data = pd.read_csv(uri)

rename_columns = {
	'unfinished' : 'finished'
}
data = data.rename(columns = rename_columns)

change_values = {
	0 : 1,
	1 : 0
}
data['finished'] = data.finished.map(change_values)

x = data[['expected_hours', 'price']]
y = data[['finished']]

raw_train_x, raw_test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.25, stratify=y)

#scale is kind of a big thing when the algorithm is learning which axys matters, 
#so we use a scaler to put things in order when we fit the model
scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model.fit(train_x, train_y)

predictions = model.predict(test_x)
accuracy = accuracy_score(test_y, predictions) * 100

print('SVC model Accuracy: %.2f' % accuracy + '%')

sns.scatterplot(x='expected_hours', y='price', hue='finished', data=data)
plt.show()