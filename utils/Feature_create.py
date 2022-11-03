import pandas as pd
import CustomImputer
from sklearn.pipeline import Pipeline

X = pd.DataFrame({'city': ['tokyo', None, 'london', 'seattle', 'san francisco', 'tokyo'],
                  'boolean': ['yes', 'no', None, 'no', 'no', 'yes'],
                  'ordinal_column': ['somewhat like', 'like', 'somewhat like', 'like', 'somewhat like', 'dislike'],
                  'quantitative_column': [1, 11, -.5, 10, None, 20]})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width', 1000)
print(X)
# print(X.isnull().sum())
print(30 * '-')


# city

cci = CustomImputer.CustomCategoryImputer(cols=['city', 'boolean'])
cci.fit_transform(X)

cqi = CustomImputer.CustomQuantitativeImputer(cols=['quantitative_column'], strategy='mean')
cqi.fit_transform(X)

imputer = Pipeline([('quant', cqi), ('category', cci)])
cd = CustomImputer.CustomDummifier(cols=['boolean','city'])
ce = CustomImputer.CustomEncoder(col='ordinal_column',ordering=['dislike', 'somewhat like','like'])
cc = CustomImputer.CustomCutter(col='quantitative_column',bins=3)

pipe = Pipeline([('imputer',imputer),('dummify',cd),('encode',ce),('cut',cc)])
pipe.fit(X)
X_trans = pipe.transform(X)
print(X_trans)