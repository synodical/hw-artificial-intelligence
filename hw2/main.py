#!/usr/bin/env python
# coding: utf-8

# In[8]:


import sys, os
sys.path.append(os.pardir)
# 부모 디렉토리에서 import할 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
# mnist data load할 수 있는 함수 import
from PIL import Image
# python image processing library
# python 버전 3.x 에서는 pillow package install해서 사


# In[9]:


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize: 0~1 실수로. 그렇지 않으면 0~255
image = x_train[0]
label = t_train[0]
# 첫번째 데이터
print(label)
print(image.shape)


# In[10]:


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    # image를 unsigned int로
    
image = image.reshape(28,28)
# 1차원 —> 2차원 (28x28)
print(image.shape)
img_show(image)


# In[11]:


import import_ipynb
from hwmyknn import MyKNeighborsClassifier

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score


# In[146]:


# input feature를 hand-crafted feature로 가공한다.
# 총 784개의 input feature는 가로 28, 세로 28개의 픽셀로 구성되어 있으므로
# 가로세로 인접한 2개의 픽셀 씩 묶어, 총 네개의 픽셀을 하나의 픽셀로 압축한다.
# 압축을 위해서, 인접한 네 개의 픽셀의 수를 더하고, 4로 나눈다.
def compress_data(data): 
    rows = 7
    cols = 7
    compr = [[0 for j in range(cols)] for i in range(rows)]
    comp = np.array(compr)
    
    for k in range(7):
        for l in range(7):
            for i in range(4*k, 4*k+4):
                for j in range(4*l, 4*l+4):
                    comp[k][l] += data[i][j]
    for k in range(7):
        for l in range(7):
            comp[k][l] /= 4
    return comp


# In[202]:


# 해당 블록의 코드는 input feature를 hand-crafted feature로 가공한다.
# 위의 compress_data function을 호출하여 data을 압축한다.
hand_crafted_train = [0 for i in range(len(x_train))]
np.array(hand_crafted_train)
for i in range(len(x_train)):
    hand_crafted_train[i] = x_train[i].reshape(28,28)

crafted_train = []
np.array(crafted_train)
for i in range(len(hand_crafted_train)):
    crafted_train.append(compress_data(hand_crafted_train[i]))
    
hand_crafted_test = [0 for i in range(len(x_test))]
np.array(hand_crafted_test)
for i in range(len(x_test)):
    hand_crafted_test[i] = x_test[i].reshape(28,28)
    
crafted_test = []
np.array(crafted_test)
for i in range(len(hand_crafted_test)):
    crafted_test.append(compress_data(hand_crafted_test[i]))


# In[234]:


# 앞선 과정에서 만들어진 crafted feature를 flatten한다.
# 즉, 아래의 과정에서 사용할 수 있도록 2차원 배열에서 1차원 배열로 차수를 줄인다.
x_crafted_test = [0 for i in range(len(x_test))]
np.array(x_crafted_test)
x_crafted_train = [0 for i in range(len(x_train))]
np.array(x_crafted_train)

for i in range(len(crafted_test)):
    x_crafted_test[i] = crafted_test[i].flatten()
for i in range(len(crafted_train)):
    x_crafted_train[i] = crafted_train[i].flatten()

print(x_crafted_test[0].shape)


# In[19]:


train_size = 1000
train_sample = np.random.randint(0, t_train.shape[0], train_size)
# 0~10000에서 숫자 10000개 골라서 train set의 index로 사용
test_size = 100
test_sample = np.random.randint(0, t_test.shape[0], test_size) 
# 0~10000에서 숫자 100개 골라서 test set의 index로 사용

''' 
X와 y는 각각 data와 target이나, 
test용 자료와 train용 자료를 _test와 _train 이라는 변수명으로 구분한다. 
X = x
y = t (실제 값)
'''
X_test = []
X_train = []
y_test = []
y_train = []
'''
아래의 x_train[i]와 x_test[i]를 crafted_train[i]와 crafted_test[i]로 바꾸면,
KNN algorithm에서 hand-crafted feature를 사용할 수 있다.
'''
for i in train_sample:
    X_train.append(x_train[i])
    y_train.append(t_train[i])

for i in test_sample:
    X_test.append(x_test[i])
    y_test.append(t_test[i])
    
X_test = np.array(X_test)
X_train = np.array(X_train)
y_test = np.array(y_test)
y_train = np.array(y_train)

label_name = ['0','1','2','3','4','5','6','7','8','9']


# In[20]:


import importlib
import hwmyknn
importlib.reload(hwmyknn)
from hwmyknn import MyKNeighborsClassifier


# In[31]:


'''Instantiate learning model (k = 13)'''
K = 15
# hwmyknn 모듈의 MyKNeighborsClassifier class를 이용해 my_classifier라는 이름의 classifier를 초기화
my_classifier = MyKNeighborsClassifier(X_test, X_train, y_test, y_train, K) 

'Test 결과를 예측하기'''
my_pred = [] # majority vote를 이용한 test 결과를 저장하는 배열
my_weighted_pred = [] # weighted majority vote를 이용한 test 결과를 저장하는 배열

for i in range(len(X_test)):
    # KNeighbors = 해당 X_test로부터 K개의 neighbors가 가진 label들
    #print(len(X_test_c)) 100
    #print(len(X_train_c)) 1000
    KNeighbors = my_classifier.get_KNeighbors(X_test[i], X_train, y_train)
    KNeighbors_with_dist = my_classifier.get_KNeighbors_with_dist(X_test[i], X_train, y_train)
    
    print(KNeighbors) # 해당 X_test로부터 K개의 neighbors가 가진 label 출력
    
    my_pred.append(my_classifier.majorityVote(KNeighbors)) # 리스트에 추가
    my_weighted_pred.append(my_classifier.weightedMajorityVote(KNeighbors_with_dist))
    


# In[32]:


print('majority vote', my_pred) # majority vote로 계산된, X_test(10개)의 예측 label
print('weighted majority vote', my_weighted_pred) # weighted majority vote로 계산된, X_test(10개)의 예측 label
print('answer', y_test)


# In[33]:


my_pred = np.array(my_pred)
my_weighted_pred = np.array(my_weighted_pred)
y_test = np.array(y_test)


# In[34]:


for i in range(len(X_test)):
    print(test_sample[i], 'th Data: result ' , label_name[my_pred[i]] , ', label: ', label_name[y_test[i]])

accuracy = accuracy_score(y_test, my_pred)*100 # 실제 label인 y값과 나의 예측값을 비교해 정확도를 계산
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')


# In[35]:


for i in range(len(X_test)):
    print(test_sample[i] , 'th Data: result ' , label_name[my_weighted_pred[i]] , ', label: ', label_name[y_test[i]])
    
accuracy = accuracy_score(y_test, my_weighted_pred)*100
print('Accuracy of our model with weighted majority vote is equal ' + str(round(accuracy, 2)) + ' %.')


# In[ ]:




