#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

class MyKNeighborsClassifier():
    # 클래스의 생성자
    def __init__(self, X_test, X_train, y_test, y_train, K=13):
        self.K=K
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test,
        self.y_train = y_train
    
    
        
    # 유클리드 거리를 구하는 함수
    def euclidian(self, data1, data2):
        dist= 0
        for i in range(len(data1)):
            dist += (data1[i] - data2[i]) ** 2 # 좌표 차의 제곱
            
        return np.sqrt(dist)
    
    # K개의 가장 가까운 거리의 이웃의 label을 구하는 함수
    def get_KNeighbors(self, X_test, X_train, y_train):
        distances = []
        targets = []
        # len(X_test) = 784, len(X_train) = 100
        # parameter로 들어오는 X_test는 1개, train은 여러개
        for j in range(len(X_train)): 
            cur = self.euclidian(X_test, X_train[j]) # cur = X_test와 X_train의 유클리드 거리
            
            distances.append((cur, y_train[j])) # distance = 거리와 해당 train 데이터의 label을 튜플로 묶은 배열
        
        sorted_distances = sorted(distances) # 오름차순으로 (거리 가까운 순) 정렬
        dist_with_label = sorted_distances[0:self.K] # 앞에서 K개의 배열을 slice
        
        for i in range(self.K): 
            targets.append(dist_with_label[i][1]) 
        # tuple의 second element(label) 만 저장한 배열 targets
        return targets
    
    # K개의 가장 가까운 거리의 이웃의 label과 그의 거리를 구하는 함수 (가중치 vote를 위해)
    def get_KNeighbors_with_dist(self, X_test, X_train, y_train):
        distances = []
        
        for j in range(len(X_train)):
            cur = self.euclidian(X_test, X_train[j])
            distances.append((cur, y_train[j]))
        
        sorted_distances = sorted(distances)
        dist_with_label = sorted_distances[0:self.K]
        # 1개의 X_test로부터 K개의 이웃 point의 거리와 label을 저장한 배열 dist_with_label
        return dist_with_label
    
    # K개의 label중 어느 것이 가장 많이 등장했는지 majority vote하는 함수
    def majorityVote(self, label):
        cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(label)):
            cnt[label[i]] += 1
        # print("majority: ", np.argmax(cnt))
        return np.argmax(cnt)
    
    # K개의 label중 어느 것의 점수가 가장 우세한지 가중치 majority vote하는 함수
    def weightedMajorityVote(self, dist_with_label):
        score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(dist_with_label)):
            # 이때 가중치는 거리의 역수로 한다(거리가 가까울수록 점수가 높다)
            score[dist_with_label[i][1]] += (dist_with_label[i][0] ** -1)
        # print("weighted majority: ", np.argmax(score))
        return np.argmax(score)


# In[4]:





# In[ ]:





# In[ ]:




