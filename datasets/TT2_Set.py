import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('data/TT_Dataset/train.csv')
test_data = pd.read_csv('data/TT_Dataset/test.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

women = train_data.loc[train_data.Sex =='female']['Survived']
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

man = train_data.loc[train_data.Sex =='male']['Survived']
rate_man = sum(man)/len(man)
print("% of man who survived:", rate_man)

y = train_data['Survived']

features = ['Pclass','Sex','SibSp','Parch']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X,y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId' : test_data.PassengerId, 'Survived':predictions})
output.to_csv('/Users/chaseli/Desktop/BA/AutoML/datasets/data/submission.csv',index=False)
print("Your submission was succesfully saved!")