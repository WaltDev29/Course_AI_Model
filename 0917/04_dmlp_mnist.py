# 은닉층을 4개로 늘린 깊은 다층 퍼셉트론(DMLP)으로 MNIST를 분류하는 예제 (5-7의 얕은 신경망과 비교)

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# MNIST 읽어 와서 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 완전연결층은 1차원 벡터만 받으므로 28x28 영상을 784개의 값으로 펼친다.
x_train = x_train.reshape(60000,784) # 텐서 모양 변환
x_test = x_test.reshape(10000,784)
# 0~255 정수를 실수로 바꾸고 255로 나눠 0~1 범위로 정규화한다.
x_train=x_train.astype(np.float32)/255.0 # ndarray로 변환
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10) # 원핫 코드로 변환
y_test=tf.keras.utils.to_categorical(y_test,10)

# 신경망 구조 설정
# 층을 지날수록 노드 수를 줄여 정보를 압축해 나가는 형태이다.
n_input=784
n_hidden1=1024
n_hidden2=512
n_hidden3=512
n_hidden4=512
n_output=10

# 신경망 구조 설계
# 5-7은 은닉층이 1개였으나 여기서는 4개이다. 층을 깊게 쌓는 것이 곧 "딥러닝"이다.
mlp=Sequential()
mlp.add(Dense(units=n_hidden1,activation='tanh',input_shape=(n_input,),kernel_initializer='random_uniform',bias_initializer='zeros'))
mlp.add(Dense(units=n_hidden2,activation='tanh',kernel_initializer='random_uniform',bias_initializer='zeros'))
mlp.add(Dense(units=n_hidden3,activation='tanh',kernel_initializer='random_uniform',bias_initializer='zeros'))
mlp.add(Dense(units=n_hidden4,activation='tanh',kernel_initializer='random_uniform',bias_initializer='zeros'))
# 출력층도 tanh를 쓰고 손실은 MSE이다. 분류 문제의 정석은 softmax와 교차 엔트로피이며, 이 차이는 5-10에서 다룬다.
mlp.add(Dense(units=n_output,activation='tanh',kernel_initializer='random_uniform',bias_initializer='zeros'))

# 신경망 학습
mlp.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
hist=mlp.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)

# 신경망의 정확률 측정
# 층을 늘린 만큼 매개변수가 크게 늘어 학습 시간도 5-7보다 오래 걸린다.
res=mlp.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)

import matplotlib.pyplot as plt

# 정확률 곡선
plt.plot(hist.history['accuracy'])      # 훈련 집합 정확률
plt.plot(hist.history['val_accuracy'])  # 검증(테스트) 집합 정확률
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.grid()
plt.show()

# 손실 함수 곡선
# 모델이 커진 만큼 과적합도 잘 일어나므로 두 곡선이 벌어지는 지점을 눈여겨본다.
plt.plot(hist.history['loss'])          # 훈련 집합 손실
plt.plot(hist.history['val_loss'])      # 검증(테스트) 집합 손실
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper right')
plt.grid()
plt.show()
