#!/usr/bin/python3

# 导入相关包
#  禁止输出warning信息
import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
from pandas import plotting


import matplotlib.pyplot as plt
plt.style.use('seaborn')

import seaborn as sns
sns.set_style("whitegrid")

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier

sns.set(style="whitegrid")

# load data
iris = sns.load_dataset('iris')


# step1, basic statistical analysis

# andrews curves
plotting.andrews_curves(iris, 'species',colormap='cool')
plt.savefig('andrews_curves.jpg')
# plt.show()

plt.close()

print(iris.info())
print(iris.head())

# set colors
antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864'] 

# draw violin plot
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True)

sns.violinplot(x='species', y='sepal_length', data=iris, palette=antV, ax=axes[0, 0])
sns.violinplot(x='species', y='sepal_width', data=iris, palette=antV, ax=axes[0, 1])
sns.violinplot(x='species', y='petal_length', data=iris, palette=antV, ax=axes[1, 0])
sns.violinplot(x='species', y='petal_width', data=iris, palette=antV, ax=axes[1, 1])
plt.savefig('violin.jpg')
# plt.show()
# plt.close()



# draw  pointplot
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True)

sns.pointplot(x='species', y='sepal_length', data=iris, color=antV[0], ax=axes[0, 0])
sns.pointplot(x='species', y='sepal_width', data=iris, color=antV[0], ax=axes[0, 1])
sns.pointplot(x='species', y='petal_length', data=iris, color=antV[0], ax=axes[1, 0])
sns.pointplot(x='species', y='petal_width', data=iris, color=antV[0], ax=axes[1, 1])
plt.savefig('pointplot.jpg')
# plt.show()
# plt.close()


#  draw pair plot

g = sns.pairplot(data=iris, palette=antV, hue= 'species')
plt.savefig('pairplot.jpg')
# plt.show()
# plt.close()


# linear regression of sepal
g = sns.lmplot(data=iris, x='sepal_width', y='sepal_length', palette=antV, hue='species')
plt.savefig('linear regression of  sepal.jpg')
# plt.show()

# linear regression of  petal
g = sns.lmplot(data=iris, x='petal_width', y='petal_length', palette=antV, hue='species')
plt.savefig('linear regression of  petal.jpg')
# plt.show()
# plt.close()

# heat map
fig=plt.gcf()
fig.set_size_inches(12, 8)
fig=sns.heatmap(iris.corr(), annot=True, cmap='GnBu', linewidths=1, linecolor='k', square=True, mask=False, vmin=-1, vmax=1, cbar_kws={"orientation": "vertical"}, cbar=True)
plt.savefig('heatmap.jpg')
# plt.show()






#  step 2 using ML
# 接下来，通过机器学习，以花萼和花瓣的尺寸为根据，预测其品种。  在进行机器学习之前，将数据集拆分为训练和测试数据集。首先，使用标签编码将 3 种鸢尾花的品种名称转换为分类值（0, 1, 2）。

# 载入特征和标签集
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris['species']

# 对标签集进行编码
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)

# 接着，将数据集以 7: 3 的比例，拆分为训练数据和测试数据：
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 101)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#  使用所有的特征，检查不同模型的准确性：
# Support Vector Machine
model = svm.SVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the SVM is: {0}'.format(metrics.accuracy_score(prediction,test_y)))

# Logistic Regression
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the Logistic Regression is: {0}'.format(metrics.accuracy_score(prediction,test_y)))

# Decision Tree
model=DecisionTreeClassifier()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the Decision Tree is: {0}'.format(metrics.accuracy_score(prediction,test_y)))

# K-Nearest Neighbours
model=KNeighborsClassifier(n_neighbors=3)
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the KNN is: {0}'.format(metrics.accuracy_score(prediction,test_y)))



# 上面使用了数据集的所有特征，下面将分别使用花瓣和花萼的尺寸：
petal = iris[['petal_length', 'petal_width', 'species']]
train_p,test_p=train_test_split(petal,test_size=0.3,random_state=0) 
train_x_p=train_p[['petal_length','petal_width']]
train_y_p=train_p.species
test_x_p=test_p[['petal_length','petal_width']]
test_y_p=test_p.species

sepal = iris[['sepal_length', 'sepal_width', 'species']]
train_s,test_s=train_test_split(sepal,test_size=0.3,random_state=0)
train_x_s=train_s[['sepal_width','sepal_length']]
train_y_s=train_s.species
test_x_s=test_s[['sepal_width','sepal_length']]
test_y_s=test_s.species

#  svm
model=svm.SVC()

model.fit(train_x_p,train_y_p) 
prediction=model.predict(test_x_p) 
print('The accuracy of the SVM using Petals is: {0}'.format(metrics.accuracy_score(prediction,test_y_p)))

model.fit(train_x_s,train_y_s) 
prediction=model.predict(test_x_s) 
print('The accuracy of the SVM using Sepal is: {0}'.format(metrics.accuracy_score(prediction,test_y_s)))

# LogisticRegression
model = LogisticRegression()

model.fit(train_x_p, train_y_p) 
prediction = model.predict(test_x_p) 
print('The accuracy of the Logistic Regression using Petals is: {0}'.format(metrics.accuracy_score(prediction,test_y_p)))

model.fit(train_x_s, train_y_s) 
prediction = model.predict(test_x_s) 
print('The accuracy of the Logistic Regression using Sepals is: {0}'.format(metrics.accuracy_score(prediction,test_y_s)))

# DecisionTreeClassifier
model=DecisionTreeClassifier()

model.fit(train_x_p, train_y_p) 
prediction = model.predict(test_x_p) 
print('The accuracy of the Decision Tree using Petals is: {0}'.format(metrics.accuracy_score(prediction,test_y_p)))

model.fit(train_x_s, train_y_s) 
prediction = model.predict(test_x_s) 
print('The accuracy of the Decision Tree using Sepals is: {0}'.format(metrics.accuracy_score(prediction,test_y_s)))

# KNN
model=KNeighborsClassifier(n_neighbors=3) 

model.fit(train_x_p, train_y_p) 
prediction = model.predict(test_x_p) 
print('The accuracy of the KNN using Petals is: {0}'.format(metrics.accuracy_score(prediction,test_y_p)))

model.fit(train_x_s, train_y_s) 
prediction = model.predict(test_x_s) 
print('The accuracy of the KNN using Sepals is: {0}'.format(metrics.accuracy_score(prediction,test_y_s)))