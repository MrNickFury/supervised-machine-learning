import pandas as pd
import graphviz
from sklearn.dummy import DummyClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

import numpy as np

uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
data = pd.read_csv(uri)
np.random.seed(8)

rename_values = {
    'yes': 1,
    'no': 0
}

data.sold = data.sold.map(rename_values)

today_year = datetime.today().year
data.model_year = today_year - data.model_year

x = data[['mileage_per_year', 'model_year', 'price']]
y = data[['sold']]

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.25, stratify=y)

# does not need a scaler to work properly, once it does not take at count the difference
# between the x values
decision_tree_model = DecisionTreeClassifier(max_depth=3)

decision_tree_model.fit(train_x, train_y)
predictions = decision_tree_model.predict(test_x)
accuracy = accuracy_score(test_y, predictions) * 100

print('DecisionTreeClassifier accuracy: %.2f' % accuracy + '%')

features = x.columns
dot_data = export_graphviz(decision_tree_model, out_file=None,
                           filled=True, rounded=True,
                           feature_names=features,
                           class_names=["sim", "nao"])
graphic = graphviz.Source(dot_data)
graphic.render(filename='Source.gv', view=True)
