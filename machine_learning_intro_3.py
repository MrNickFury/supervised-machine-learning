from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

model = LinearSVC()
uri='https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'
data = pd.read_csv(uri) #returns a two-dimensional data structure with labeled axes.

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

SEED = 18

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.25, random_state=SEED, stratify=y)

model.fit(train_x, train_y)

predictions = model.predict(test_x)
accuracy = accuracy_score(test_y, predictions) * 100

#returns a low accuracy, because it can only predict analising linear data, but if we
#take a look at the plots we notice that our data relationship is non-linear
print('LinearSVC model Accuracy: %.2f' % accuracy + '%')

sns.relplot(x='expected_hours', y='price', hue='finished', col='finished', data=data)
plt.show()