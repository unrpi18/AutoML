import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
df = pd.read_csv('data/ARSCMA_Dataset/1.csv', header=None)
df.columns = ['index', 'x', 'y', 'z', 'activity']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

print(df.head())
# basic accuracy= 0.515369
print(df['activity'].value_counts(normalize=True))

X = df[['x', 'y', 'z']]
y = df['activity']

knn = KNeighborsClassifier()

# grid.fit(X,y)
# accuracy = 0.736
# print(grid.best_score_, grid.best_params_)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

pipe_params = {'poly_features__degree':[1,2,3],'poly_features__interaction_only': [True,False],
               'classify__n_neighbors':[3,4,5,6]}

pipe = Pipeline([('poly_features',poly),('classify',knn)])
grid = GridSearchCV(pipe,pipe_params)
grid.fit(X,y)

# accuracy = 0.739786
print(grid.best_score_, grid.best_params_)
print("ok")