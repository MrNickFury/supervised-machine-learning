from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#pig = 0, dog = 1
#pelo longo, perna curta, auau --> 0 = no | 1 = yes
pig1 = [1, 1, 0]
pig2 = [0, 1, 0]
pig3 = [0, 1, 1]

dog1 = [1, 1, 1]
dog2 = [1, 0, 1]
dog3 = [0, 1, 1]

train_x = [pig1, pig2, pig3, dog1, dog2, dog3]
train_y = [0, 0, 0, 1, 1, 1]

model = LinearSVC()
model.fit(train_x, train_y)

misterious_animal1 = [1, 1, 1]
misterious_animal2 = [1, 1, 0]
misterious_animal3 = [0, 1, 1]

teste_x = [misterious_animal1, misterious_animal2, misterious_animal3]
teste_y = [1, 0, 0]

predictions = model.predict(teste_x)

accuracy = accuracy_score(teste_y, predictions)

print('LinearSVC model Accuracy: %.2f' % (accuracy * 100) + '%')