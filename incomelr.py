import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix


data=pd.read_csv("income.csv")
cpdata=data.copy(deep=True)

def checkunique(s):
    print(np.unique(cpdata[s]))


print(cpdata.info())
''' All Data Type Ok'''

print(cpdata.isnull().sum())
'''No Null Values'''

desc=cpdata.describe()
desc_obj=cpdata.describe(include="O")


checkunique("occupation")
checkunique("JobType")

cpdata=pd.read_csv("income.csv",na_values=[" ?"])
print(cpdata.info())
print(cpdata.isnull().sum())
'''New missing data'''

#examine missing data
missing=cpdata[cpdata.isnull().any(axis=1)]

cpdata.dropna(axis=0,inplace=True)
print(cpdata.info())

cordata=cpdata.corr()
plt=sns.pairplot(cpdata,hue="SalStat")

comp=pd.crosstab(index=cpdata["SalStat"],columns=cpdata["gender"],normalize=True,margins=True)

'''Visualizing salstat'''
var=sns.countplot(x=cpdata["SalStat"])
for p in var.patches:
   var.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))

'''Age hist'''
sns.histplot(data["age"],bins=10)

sns.boxplot(y=cpdata["age"],x=cpdata["SalStat"])
cpdata.groupby("SalStat")["age"].median()

h=sns.countplot(y=cpdata["EdType"],hue=cpdata["SalStat"])

x=pd.crosstab(index=cpdata["JobType"],columns=cpdata["SalStat"],margins=True,normalize='index')
x=x*100

sns.histplot(x="capitalloss",data=cpdata)

sns.boxplot(y="hoursperweek",x="SalStat",data=cpdata) 
'''give oredr arg'''

cpdata["SalStat"]=cpdata["SalStat"].map({" less than or equal to 50,000":0," greater than 50,000":1})

newdata=cpdata.drop(['gender','nativecountry','race','JobType'],axis=1)
newdata=pd.get_dummies(newdata,drop_first=True)

col_list=list(newdata.columns)
ip_list=list(set(col_list)-set(["SalStat"]))
y=newdata["SalStat"].values
x=newdata[ip_list].values

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size= 0.3,random_state=0)

logistic=LogisticRegression()

logistic.fit(train_x,train_y)
prediction=logistic.predict(test_x)

confusion_mat=confusion_matrix(test_y,prediction)
accuracy=accuracy_score(test_y,prediction)
print("misinterpreted values count : ",(test_y != prediction).sum())
