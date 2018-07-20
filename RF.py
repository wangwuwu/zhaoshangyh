#coding=utf-8

#coding=utf-8
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
from numpy import *
from sklearn import  metrics
from sklearn.decomposition import PCA
from random import sample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# train_agg=pd.read_table('train_agg.csv',header=0,sep='	')
# train_flg=pd.read_table('train_flg.csv',header=0,sep='	')
# data=pd.merge(train_agg,train_flg,on="USRID",how='inner')

def buildForest(data,n,i,j):
    '''

    :param data:数据集
    :param n: 树的个数
    :param i: 训练每棵树数据量占总数的百分比
    :param j: 训练每棵树的属性树占总属性树的百分比
    :return: n棵树的数组
    '''
    res=[]
    label=data['FLAG'].values
    del data['FLAG']
    data=array(data)
    a,b=shape(data)
    for k in range(n):
        indexs=sample(range(a),int(a*i))
        columns=sample(range(b),int(b*j)).sort()
        train_data=data[indexs,:][:,columns]
        train_label=label[indexs]
        dc=DecisionTreeClassifier(class_weight='balanced',min_samples_leaf=6,min_samples_split=10)
        dc.fit(train_data,train_label)
        res.append(dc)
    return res

def ForesPredict(fr,dataNode):
    #计算数据点属于类1的概率
    res=[]
    for tree in fr:
        res.append(tree.predict_proba(dataNode))
    return sum(res)/len([i for i in res if i>0])#

def ForesPredicts(fr,dataNodes):
    res=[]
    for i in dataNodes:
        temp=ForesPredict(fr,i)
        res.append(temp)
    return res

data=pd.read_csv('train.csv',header=0,sep=',')
data.fillna(0,inplace=True)
print(shape(data))
splitPoint=75000#划分训练集测试集

del data['USRID']

train_label=data.ix[0:splitPoint,'FLAG'].values
test_label=data.ix[splitPoint:,'FLAG'].values

del data['FLAG']
pca=PCA(n_components=100)
pca.fit(data)
data=pca.transform(data)
train_data=data[:splitPoint+1,:]
test_data=data[splitPoint:,:]

# train_data=data.ix[0:splitPoint,:].drop('FLAG',axis=1).values
# test_data=data.ix[splitPoint:,:].drop('FLAG',axis=1).values
# del test_data['FLAG']
# del train_data['FLAG']

rfc=RandomForestClassifier(n_estimators=51,min_samples_split=10,min_samples_leaf=10,class_weight='balanced',max_depth=10)
# #建立随机森林
# forest=buildForest(train_data,151,0.7,0.7)
# #对测试集预测
# test_y=ForesPredicts(test_data)
rfc.fit(train_data,train_label)
res=rfc.predict_proba(test_data)
test_y=res[:,1]


pca1=PCA(n_components=50)
pca.fit(data)
data=pca.transform(data)
train_data=data[:splitPoint+1,:]
test_data=data[splitPoint:,:]
lr=LogisticRegression(penalty='l1',class_weight='balanced')
lr.fit(train_data,train_label)
res1=lr.predict_proba(test_data)
test_y1=res1[:,1]
test_yy=[0.8*test_y1[i]+0.3*test_y[i] for i in range(len(test_y))]
#test_yy=sum([test_y,test_y1],axis=0)

np.save('rf_result_'+str(metrics.roc_auc_score(test_label,test_y))+'.npy',test_y)
np.save('lr_result_'+str(metrics.roc_auc_score(test_label,test_y1))+'.npy',test_y1)


#计算auc
result=metrics.roc_auc_score(test_label,test_yy)
print(result)