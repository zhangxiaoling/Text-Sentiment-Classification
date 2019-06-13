#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 20:43:20 2019

@author: zxl
"""

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
import pickle
import pandas as pd

df=pd.read_csv('train.csv',lineterminator='\n')
print(df.head())

print(len(df['label']))
print(type(df['label']))

df_test=pd.read_csv('test.csv',lineterminator='\n')
print(len(df_test['review']))
print(df_test.head())

def load_data(type='train'):
    if type=='train':
        data=[review.lower() for review in df['review']]
        labels=[1 if label=='Positive' else 0 for label in df['label']]
        return data,labels
    else:
        data=[review.lower() for review in df_test['review']]
        return data
    
def train_tfidf(train_data):
    tfidf=TFIDF(min_df=5,max_features=5000,ngram_range=(1,3),use_idf=1,smooth_idf=1)
    tfidf.fit(train_data)
    return tfidf

def train_SVC(data_vec,label):
    svc=LinearSVC()
    clf=CalibratedClassifierCV(svc)
    clf.fit(data_vec,label)
    return clf

def train():
    train_data,labels=load_data('train')
    tfidf=train_tfidf(train_data)
    train_vec=tfidf.transform(train_data)
    model=train_SVC(train_vec,labels)
    print('model saving...')
    joblib.dump(tfidf,'tfidf.model')
    joblib.dump(model,'svc.model')
    
def predict():
    test_data=load_data('test')
    print('load model...')
    tfidf=joblib.load('tfidf.model')
    model=joblib.load('svc.model')
    test_vec=tfidf.transform(test_data)
    test_predict=model.predict_proba(test_vec)
    return test_predict

def train_no_save_model():
    train_data,labels=load_data('train')
    tfidf=train_tfidf(train_data)
    train_vec=tfidf.transform(train_data)
    model=train_SVC(train_vec,labels)
    test_data=load_data('test')
    test_vec=tfidf.transform(test_data)
    test_predict=model.predict_proba(test_vec)
    return test_predict

train()
test_predict=predict()
test_predict_positive=[item[1] for item in test_predict]
print(test_predict[:5])

test_ids=df_test['ID']
Data={'ID':test_ids,'Pred':test_predict_positive}
pd.DataFrame(Data,columns=['ID','Pred']).to_csv('test_pred.csv',header=True,index=False)
print('Done')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

