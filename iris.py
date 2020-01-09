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
# 搞一稿每个图的可视化如何做的，然后作用是什么，缺点是什么，写到每个图代码注释里边

# andrews curves 又名 adrews plot
#  rolled-down version, non-integer version of KK radar m chart
# or smoothed version of parallel coordinate plot
# 横轴是theta相位，竖轴是映射函数的值
# andrews 是将每个样本全部纬度数据映射到一个函数上，
# 这个函数定义如下，fx(theta) = xi(1)/sqrt(2) + xi(2)*sin(theta)+xi(3)*sin(2*theta)+xi(4)*sin(3*theta)+ ....
# andrews method 可以很好的保留均值、方差、不同类别样本的距离；这样可以帮助我们观察如果曲线挨的特别近，则可能是同一个类别的，这一点在iris数据集上有很好的证明；
# 自己写一个andrew curves，纬度暂时固定

def alex_andrews_curves(x,theta): # 注意这里x，theta 必须均为numpy中的数组
    # 因为iris有4个纬度，因此映射函数写成这个样子
    base_functions = [lambda x: x[0]/np.sqrt(2.), lambda x: x[1]*np.sin(theta), lambda x: x[2]*np.cos(theta), lambda x: x[3]*np.sin(2.*theta)]  
    curve = np.zeros(len(theta))
    for f in base_functions:
        curve += f(x)
    return curve
iris_samples = np.loadtxt('iris_noHeader.csv', usecols=[0,1,2,3], delimiter=',')
iris_classes = np.loadtxt('iris_noHeader.csv', usecols=[4], delimiter=',', dtype=np.str)
theta = np.linspace(-np.pi, np.pi, 100)
for s in iris_samples[0:50]:
    plt.plot(theta, alex_andrews_curves(s, theta), 'r')
for s in iris_samples[51:100]:
    plt.plot(theta, alex_andrews_curves(s, theta), 'g')
for s in iris_samples[101:150]:
    plt.plot(theta, alex_andrews_curves(s, theta), 'b')
    
plt.title('alex self-made andrews curves')
# plt.show()

plotting.andrews_curves(iris, 'species',colormap='cool')
plt.savefig('andrews_curves.jpg')
# plt.show()

# plt.close()

print(iris.info())
print(iris.head())
print(iris['species'].value_counts())

# scatter plot
# 散点图的横轴是petal_length， 竖轴是petal_width，缺点是只能展现2D数据的直接分布情况
plt.scatter( iris['petal_length'], iris['petal_width'], c='blue',s=40 )
# plt.show()

# joint plot，散点图与直方图的合并
sns.jointplot(x='sepal_length', y = 'sepal_width', data=iris, size=5)
# plt.show()


# use seaborn's FacetGrid to color the scatterplot by species，这个是按照类别加以不同的颜色的散点图
sns.FacetGrid(iris, hue='species', size=5).map(plt.scatter,'sepal_length', 'sepal_width').add_legend()
# plt.show()

# boxplot 特点是不受异常值的影响，分位数的计算公式如下：Qi = (n+1)/4*i, IQR[inter-quartile range] = Q3-Q1, min = Q1-1.5*IQR; max = Q3+1.5*IQR;
sns.boxplot(x='species', y='sepal_length', data=iris)
# plt.show()
sns.boxplot(x='species', y='sepal_width', data=iris)
# plt.show()
sns.boxplot(x='species', y='petal_length', data=iris)
# plt.show()
sns.boxplot(x='species', y='petal_width', data=iris)
# plt.show()

# strip plot, 散点图与箱线图的合并
sns.boxplot(x='species', y='sepal_length', data=iris)
sns.stripplot(x='species', y='sepal_length', data=iris, jitter= True, edgecolor='gray')
# plt.show()

# set colors
antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864'] 

# draw violin plot，竖轴表示，横轴表示，提琴图综合箱线图与密度图的优点，中间白点表示中位数，黑色粗线上下分别表示75%、25%分位数，黑色细线的上下分别表示max，min，超出黑色细线[叫做须]范围的为异常点；黑色粗线+黑色细线总体表示95%置信区间【为何？】；可以观察概率密度分布是单峰[偏态分布哪边少是哪边偏，注意偏态分布中三个指标参数：均值、众数、中位数]，双峰，多峰【unimodal, bimodal还是multimodal，分布多峰，双峰意味着某个随机变量的分布中两个高频被一个低频分割开；】外部形状为核密度估计，在概率论中用来估计未知的密度函数，属于非参数检验方法之一；
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

# kde plot , creates and visualizes a kernel density estimate of the underlying feature，kde图是核密度图【kernel density estimation】，作用是给定一个样本集，输出样本的分布密度函数，解决这个问题有2个办法，第一个是参数估计法，根据先验知识假定随机变量的分布模型，然后使用数据集去拟合模型的参数；第二个方法是非参数估计法，  
sns.FacetGrid(iris, hue='species', size=6).map(sns.kdeplot, 'sepal_width').add_legend()
# plt.show()


# 直方图
n,bins, patches = plt.hist(iris['sepal_length'])
plt.title('alex histogram')
plt.show()



# parallel_coordinates
# Parallel coordinates plots each feature on a separate column & then draws lines
# connecting the features for each data sample
from pandas.plotting import parallel_coordinates
parallel_coordinates(iris, 'species')
# plt.show()

# radviz plot
from pandas.plotting import radviz
radviz(iris, 'species')
plt.show()


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


# 检查利用数据集是否线性可分与  类别凸包交集为空这一充要条件，计算 数据集的凸包 然后check数据集的线性可分性，为模型的选择提供依据，【奥卡姆  剃刀原则】

import scipy.spatial.ConvexHull


#  总结
# iris 数据集是英国生物学家观测鸢尾花花瓣、花萼后得出的宽度、长度数据；样本数150，类别数3，特征维度4；总体上来说是线性
