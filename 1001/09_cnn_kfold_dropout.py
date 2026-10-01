# k-겹 교차 검증으로 드롭아웃이 정확률을 올리는지 확인하는 예제

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
os.environ["KERAS_BACKEND"]="torch"

# 지금 CPU로 도는지 GPU로 도는지 출력한다
# 위의 백엔드 지정이 적용되지 않으면 여기서 드러난다
import keras,torch
print("백엔드:",keras.backend.backend(),"/ 장치:","GPU "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.optimizers import Adam
# Keras 3에서 ImageDataGenerator는 legacy 모듈로 이동하였다
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

# 하이퍼 매개변수 설정
# 한 번 실행에 신경망을 k번(=5번) 학습하므로 시간이 오래 걸린다
batch_siz=128
n_epoch=10
k=5 # k-겹 교차 검증

# 드롭아웃 비율에 따라 교차 검증을 수행하고 정확률을 반환하는 함수
# 훈련 집합을 k등분한 뒤 한 덩어리씩 돌아가며 검증에 쓰고 나머지로 학습한다
# 한 번만 재면 우연이 섞이지만 k번 재면 성능 차이를 믿을 수 있다
def cross_validation(dropout_rate):
    accuracy=[]
    for train_index,val_index in KFold(k).split(x_train):
        # 훈련 집합과 검증 집합으로 분할
        xtrain,xval=x_train[train_index],x_train[val_index]
        ytrain,yval=y_train[train_index],y_train[val_index]

        # 신경망 모델 설계
        # 겹마다 반드시 새 모델을 만들어야 한다. 그러지 않으면 앞 겹에서
        # 학습한 가중치가 남아 검증 집합을 이미 본 셈이 되기 때문이다
        cnn=Sequential()
        cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
        cnn.add(Conv2D(32,(3,3),activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2,2)))
        cnn.add(Dropout(dropout_rate[0]))  # 비율 0.0이면 드롭아웃을 끈 것과 같다
        cnn.add(Conv2D(64,(3,3),activation='relu'))
        cnn.add(Conv2D(64,(3,3),activation='relu'))
        cnn.add(MaxPooling2D(pool_size=(2,2)))
        cnn.add(Dropout(dropout_rate[1]))
        cnn.add(Flatten())
        cnn.add(Dense(512,activation='relu'))
        cnn.add(Dropout(dropout_rate[2]))
        cnn.add(Dense(10,activation='softmax'))

        # 신경망 모델을 학습하고 평가하기
        cnn.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
        cnn.fit(xtrain,ytrain,batch_size=batch_siz,epochs=n_epoch,verbose=1)
        accuracy.append(cnn.evaluate(xval,yval,verbose=1)[1])
    return accuracy  # 겹마다의 정확률 k개를 모아 돌려준다

# 드롭아웃 비율을 달리하며 신경망을 평가
# 구조는 같고 드롭아웃 비율만 다르므로 차이는 드롭아웃 때문이라고 볼 수 있다
acc_without_dropout=cross_validation([0.0,0.0,0.0])
acc_with_dropout=cross_validation([0.25,0.25,0.5])

print("드롭아웃 적용 안 할 때:",np.array(acc_without_dropout).mean())
print("드롭아웃 적용할 때:",np.array(acc_with_dropout).mean())

import matplotlib.pyplot as plt

# 박스 플롯으로 정확률 표시
# 평균만 보지 않고 k개 값의 흩어진 정도까지 함께 보기 위한 그림이다
# 두 상자가 겹치지 않아야 차이가 의미 있다고 말할 수 있다
plt.grid()
plt.boxplot([acc_without_dropout,acc_with_dropout],tick_labels=["Without Dropout","With Dropout"])
plt.show()