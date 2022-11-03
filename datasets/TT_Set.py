import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns
import warnings

from utils import CustomImputer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold
warnings.filterwarnings('ignore')

train_data = pd.read_csv('data/TT_Dataset/train.csv')
test_data = pd.read_csv('data/TT_Dataset/test.csv')
full = train_data.append(test_data, ignore_index=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

sns.set_style('whitegrid')
print(train_data.head())

# accuracy = 0.6161(0), 0.3838(1)
print(train_data['Survived'].value_counts(normalize=True))

print(train_data.isnull().sum())
print(30 * '-')
# embarked
cci = CustomImputer.CustomCategoryImputer(cols=['Embarked'])
train_data_cci = cci.fit_transform(train_data)
print(train_data_cci.isnull().sum())
print(30 * '-')

# 对Cabin缺失值进行处理，利用U（Unknown）填充缺失值
full['Cabin'] = full['Cabin'].fillna('U')

full['Fare'] = full['Fare'].fillna(
    full[(full['Pclass'] == 3) & (full['Embarked'] == 'S') & (full['Cabin'] == 'U')]['Fare'].mean())
full['Title'] = full['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
print(30 * '-')
print(full['Title'].value_counts())

TitleDict = {'Mr': 'Mr', 'Mlle': 'Miss', 'Miss': 'Miss', 'Master': 'Master', 'Jonkheer': 'Master', 'Mme': 'Mrs',
             'Ms': 'Mrs', 'Mrs': 'Mrs', 'Don': 'Royalty', 'Sir': 'Royalty', 'the Countess': 'Royalty',
             'Dona': 'Royalty',
             'Lady': 'Royalty', 'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer', 'Dr': 'Officer',
             'Rev': 'Officer'}

full['Title'] = full['Title'].map(TitleDict)
print(30 * '-')
print(full['Title'].value_counts())
sns.barplot(data=full, x='Title', y='Survived')

full['familyNum'] = full['Parch'] + full['SibSp'] + 1
sns.barplot(data=full, x='familyNum', y='Survived')



def familysize(familyNum):
    if familyNum == 1:
        return 0
    elif (familyNum >= 2) & (familyNum <= 4):
        return 1
    else:
        return 2


full['familySize'] = full['familyNum'].map(familysize)
print(full['familySize'].value_counts())

sns.barplot(data=full,x='familySize',y='Survived')


#提取Cabin字段首字母
full['Deck']=full['Cabin'].map(lambda x:x[0])
#查看不同Deck类型乘客的生存率
sns.barplot(data=full,x='Deck',y='Survived')



TickCountDict = full['Ticket'].value_counts()
full['TickCot']=full['Ticket'].map(TickCountDict)
print(full['TickCot'].head())
sns.barplot(data=full,x='TickCot',y='Survived')

def TickCountGroup(num):
    if (num>=2)&(num<=4):
        return 0
    elif (num==1)|((num>=5)&(num<=8)):
        return 1
    else :
        return 2


# 得到各位乘客TickGroup的类别
full['TickGroup'] = full['TickCot'].map(TickCountGroup)

print(full.head())
AgePre=full[['Age','Parch','Pclass','SibSp','Title','familyNum','TickCot']]
AgePre=pd.get_dummies(AgePre)
ParAge=pd.get_dummies(AgePre['Parch'],prefix='Parch')
SibAge=pd.get_dummies(AgePre['SibSp'],prefix='SibSp')
PclAge=pd.get_dummies(AgePre['Pclass'],prefix='Pclass')
AgeCorrDf = AgePre.corr()
print(AgeCorrDf['Age'].sort_values())
AgePre=pd.concat([AgePre,ParAge,SibAge,PclAge],axis=1)
AgeKnown=AgePre[AgePre['Age'].notnull()]
AgeUnKnown=AgePre[AgePre['Age'].isnull()]
AgeKnown_X=AgeKnown.drop(['Age'],axis=1)
AgeKnown_y=AgeKnown['Age']
AgeUnKnown_X=AgeUnKnown.drop(['Age'],axis=1)
rfr=RandomForestRegressor(random_state=None,n_estimators=500,n_jobs=-1)
rfr.fit(AgeKnown_X,AgeKnown_y)

print(rfr.score(AgeKnown_X,AgeKnown_y))

AgeUnKnown_y=rfr.predict(AgeUnKnown_X)
full.loc[full['Age'].isnull(),['Age']]=AgeUnKnown_y
print(full.info())

#提取乘客的姓氏及相应的乘客数
full['Surname']=full['Name'].map(lambda x:x.split(',')[0].strip())

SurNameDict=full['Surname'].value_counts()
full['SurnameNum']=full['Surname'].map(SurNameDict)

#将数据分为两组
MaleDf=full[(full['Sex']=='male')&(full['Age']>12)&(full['familyNum']>=2)]
FemChildDf=full[((full['Sex']=='female')|(full['Age']<=12))&(full['familyNum']>=2)]

MSurNamDf=MaleDf['Survived'].groupby(MaleDf['Surname']).mean()
MSurNamDf.head()
print(MSurNamDf.value_counts())

MSurNamDict=MSurNamDf[MSurNamDf.values==1].index
print(MSurNamDict)
FCSurNamDf=FemChildDf['Survived'].groupby(FemChildDf['Surname']).mean()
FCSurNamDict=FCSurNamDf[FCSurNamDf.values==0].index

#对数据集中这些姓氏的男性数据进行修正：1、性别改为女；2、年龄改为5。
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(MSurNamDict))&(full['Sex']=='male'),'Age']=5
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(MSurNamDict))&(full['Sex']=='male'),'Sex']='female'

#对数据集中这些姓氏的女性及儿童的数据进行修正：1、性别改为男；2、年龄改为60。
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurNamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Age']=60
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurNamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Sex']='male'
#人工筛选
fullSel=full.drop(['Cabin','Name','Ticket','PassengerId','Surname','SurnameNum'],axis=1)
#查看各特征与标签的相关性
corrDf=fullSel.corr()
print(corrDf['Survived'].sort_values(ascending=True))

#热力图，查看Survived与其他特征间相关性大小
plt.figure(figsize=(8,8))
sns.heatmap(fullSel[['Survived','Age','Embarked','Fare','Parch','Pclass',
                     'Sex','SibSp','Title','familyNum','familySize','Deck',
                     'TickCot','TickGroup']].corr(),cmap='BrBG',annot=True,
            linewidths=.5)
plt.xticks(rotation=45)
plt.show()
fullSel=fullSel.drop(['familyNum','SibSp','TickCot','Parch'],axis=1)
#one-hot编码
fullSel=pd.get_dummies(fullSel)
PclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')
TickGroupDf=pd.get_dummies(full['TickGroup'],prefix='TickGroup')
familySizeDf=pd.get_dummies(full['familySize'],prefix='familySize')

fullSel=pd.concat([fullSel,PclassDf,TickGroupDf,familySizeDf],axis=1)
print(fullSel.head())

experData=fullSel[fullSel['Survived'].notnull()]
preData=fullSel[fullSel['Survived'].isnull()]

experData_X=experData.drop('Survived',axis=1)
experData_y=experData['Survived']
preData_X=preData.drop('Survived',axis=1)




#设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=10)

#汇总不同模型算法
classifiers= [SVC(), DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier(),
              GradientBoostingClassifier(), KNeighborsClassifier(), LogisticRegression(), LinearDiscriminantAnalysis()]

cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier,experData_X,experData_y,
                                      scoring='accuracy',cv=kfold,n_jobs=-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

# 汇总数据
cvResDf = pd.DataFrame({'cv_mean': cv_means,
                        'cv_std': cv_std,
                        'algorithm': ['SVC', 'DecisionTreeCla', 'RandomForestCla', 'ExtraTreesCla',
                                      'GradientBoostingCla', 'KNN', 'LR', 'LinearDiscrimiAna']})
print(cvResDf)

cvResFacet=sns.FacetGrid(cvResDf.sort_values(by='cv_mean',ascending=False),sharex=False,
                         sharey=False,aspect=2)
cvResFacet.map(sns.barplot,'cv_mean','algorithm',**{'xerr':cv_std},
               palette='muted')
cvResFacet.set(xlim=(0.7,0.9))
cvResFacet.add_legend()
plt.show()

#GradientBoostingClassifier模型
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["log_loss"],
                 'n_estimators' : [100,200,300],
                 'learning_rate': [0.1, 0.05, 0.01],
                 'max_depth': [4, 8],
                 'min_samples_leaf': [100,150],
                 'max_features': [0.3, 0.1]
                 }
modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold,
                          scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsGBC.fit(experData_X,experData_y)

#LogisticRegression模型
modelLR=LogisticRegression()
LR_param_grid = {'C' : [1,2,3],
                 'penalty':['l1','l2']}
modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold,
                         scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsLR.fit(experData_X,experData_y)
print('modelgsGBC模型得分为：%.3f'%modelgsGBC.best_score_)
#modelgsLR模型
print('modelgsLR模型得分为：%.3f'%modelgsLR.best_score_)

#TitanicGBSmodle
GBCpreData_y=modelgsGBC.predict(preData_X)
GBCpreData_y=GBCpreData_y.astype(int)
#导出预测结果
GBCpreResultDf=pd.DataFrame()
GBCpreResultDf['PassengerId']=full['PassengerId'][full['Survived'].isnull()]
GBCpreResultDf['Survived']=GBCpreData_y
print(GBCpreResultDf)
GBCpreResultDf.to_csv('data/TitanicGBSmodle.csv',index=False)