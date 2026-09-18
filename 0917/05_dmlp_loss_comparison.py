# 손실 함수를 평균제곱오차와 교차 엔트로피로 바꿔가며 성능을 비교하는 예제 (구조와 옵티마이저는 동일)

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# MNIST 읽어 와서 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,784)                # 28x28 영상을 784차원 벡터로 펼침
x_test = x_test.reshape(10000,784)
x_train=x_train.astype(np.float32)/255.0            # 0~255를 0~1로 정규화
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)   # 정답을 원핫 코드로 변환
y_test=tf.keras.utils.to_categorical(y_test,10)

# 신경망 구조 설정
n_input=784
n_hidden1=1024
n_hidden2=512
n_hidden3=512
n_hidden4=512
n_output=10

# 평균제곱오차를 사용한 모델
# 5-9와 달리 출력층 활성 함수가 softmax이다. softmax는 10개 출력의 합이 1이 되게 만들어 확률처럼 해석할 수 있게 한다.
dmlp_mse=Sequential()
dmlp_mse.add(Dense(units=n_hidden1,activation='tanh',input_shape=(n_input,)))
dmlp_mse.add(Dense(units=n_hidden2,activation='tanh'))
dmlp_mse.add(Dense(units=n_hidden3,activation='tanh'))
dmlp_mse.add(Dense(units=n_hidden4,activation='tanh'))
dmlp_mse.add(Dense(units=n_output,activation='softmax'))
# 손실 함수만 mean_squared_error 이다. 아래 모델과의 유일한 차이가 바로 이 부분이다.
dmlp_mse.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])
hist_mse=dmlp_mse.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)

# 교차 엔트로피를 사용한 모델
# 구조, 옵티마이저, 학습률, 세대 수를 모두 위와 똑같이 맞춰야 손실 함수의 효과만 따로 볼 수 있다.
dmlp_ce=Sequential()
dmlp_ce.add(Dense(units=n_hidden1,activation='tanh',input_shape=(n_input,)))
dmlp_ce.add(Dense(units=n_hidden2,activation='tanh'))
dmlp_ce.add(Dense(units=n_hidden3,activation='tanh'))
dmlp_ce.add(Dense(units=n_hidden4,activation='tanh'))
dmlp_ce.add (Dense(units=n_output,activation='softmax'))
# 교차 엔트로피는 틀린 정도가 클수록 손실을 더 크게 주기 때문에 분류 문제에서 학습이 빠르고 정확률도 높다.
dmlp_ce.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])
hist_ce=dmlp_ce.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)

# 두 모델의 정확률 비교
res_mse=dmlp_mse.evaluate(x_test,y_test,verbose=0)
print("평균제곱오차의 정확률은",res_mse[1]*100)
res_ce=dmlp_ce.evaluate(x_test,y_test,verbose=0)
print("교차 엔트로피의 정확률은",res_ce[1]*100)

# 하나의 그래프에서 두 모델을 비교
# 네 곡선을 겹쳐 그려 손실 함수에 따라 학습 속도가 어떻게 달라지는지 확인한다.
import matplotlib.pyplot as plt
plt.plot(hist_mse.history['accuracy'])      # MSE 모델의 훈련 정확률
plt.plot(hist_mse.history['val_accuracy'])  # MSE 모델의 검증 정확률
plt.plot(hist_ce.history['accuracy'])       # 교차 엔트로피 모델의 훈련 정확률
plt.plot(hist_ce.history['val_accuracy'])   # 교차 엔트로피 모델의 검증 정확률
plt.title('Model accuracy comparison between MSE and cross entropy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train_mse','Validation_mse','Train_ce','Validation_ce'], loc='best')
plt.grid()
plt.show()
