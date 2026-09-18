# 옵티마이저 4종(SGD, Adam, Adagrad, RMSprop)의 성능을 같은 조건에서 비교하는 예제

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD,Adam,Adagrad,RMSprop

# fashion MNIST 읽어 와서 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
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

# 하이퍼 매개변수 설정
# 네 모델이 똑같은 조건에서 겨루도록 값을 한곳에 모아 두었다.
batch_siz=256
n_epoch=50

# 모델을 설계해주는 함수(모델을 나타내는 객체 model을 반환)
# 같은 코드를 네 번 쓰지 않으려고 함수로 묶었다. 호출할 때마다 가중치가 새로 초기화된 모델이 만들어진다.
def build_model():
    model=Sequential()
    # 5-9, 5-10의 tanh와 달리 relu를 쓴다. relu는 깊은 신경망에서 기울기가 사라지는 문제가 덜해 학습이 잘 된다.
    model.add(Dense(units=n_hidden1,activation='relu',input_shape=(n_input,)))
    model.add(Dense(units=n_hidden2,activation='relu'))
    model.add(Dense(units=n_hidden3,activation='relu'))
    model.add(Dense(units=n_hidden4,activation='relu'))
    model.add(Dense(units=n_output,activation='softmax'))   # 출력층은 확률 형태로 만드는 softmax
    return model

# SGD 옵티마이저를 사용하는 모델
# 가장 기본적인 경사 하강법이다. 괄호를 비웠으므로 기본 학습률(0.01)이 쓰인다.
dmlp_sgd=build_model()
dmlp_sgd.compile(loss='categorical_crossentropy',optimizer=SGD(),metrics=['accuracy'])
hist_sgd=dmlp_sgd.fit(x_train,y_train,batch_size=batch_siz,epochs=n_epoch,validation_data=(x_test,y_test),verbose=2)

# Adam 옵티마이저를 사용하는 모델
# 관성과 학습률 자동 조절을 함께 쓰는 기법으로, 실무에서 가장 널리 쓰인다.
dmlp_adam=build_model()
dmlp_adam.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
hist_adam=dmlp_adam.fit(x_train,y_train,batch_size=batch_siz,epochs=n_epoch,validation_data=(x_test,y_test),verbose=2)

# Adagrad 옵티마이저를 사용하는 모델
# 매개변수마다 학습률을 다르게 준다. 다만 학습이 진행될수록 학습률이 계속 작아지는 성질이 있다.
dmlp_adagrad=build_model()
dmlp_adagrad.compile(loss='categorical_crossentropy',optimizer=Adagrad(),metrics=['accuracy'])
hist_adagrad=dmlp_adagrad.fit(x_train,y_train,batch_size=batch_siz,epochs=n_epoch,validation_data=(x_test,y_test),verbose=2)

# RMSprop 옵티마이저를 사용하는 모델
# Adagrad의 학습률이 지나치게 줄어드는 단점을 보완한 기법이다.
dmlp_rmsprop=build_model()
dmlp_rmsprop.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
hist_rmsprop=dmlp_rmsprop.fit(x_train,y_train,batch_size=batch_siz,epochs=n_epoch,validation_data=(x_test,y_test),verbose=2)

# 네 모델의 정확률을 출력
# evaluate가 돌려주는 [손실, 정확률] 중 두 번째 값만 꺼내 쓴 것이다.
print("SGD 정확률은",dmlp_sgd.evaluate(x_test,y_test,verbose=0)[1]*100)
print("Adam 정확률은",dmlp_adam.evaluate(x_test,y_test,verbose=0)[1]*100)
print("Adagrad 정확률은",dmlp_adagrad.evaluate(x_test,y_test,verbose=0)[1]*100)
print("RMSprop 정확률은",dmlp_rmsprop.evaluate(x_test,y_test,verbose=0)[1]*100)

import matplotlib.pyplot as plt

# 네 모델의 정확률을 하나의 그래프에서 비교
# 색으로 옵티마이저를 구분하고, 실선은 훈련 집합, 점선(--)은 검증 집합을 뜻한다.
plt.plot(hist_sgd.history['accuracy'],'r')          # 빨강 실선: SGD 훈련
plt.plot(hist_sgd.history['val_accuracy'],'r--')    # 빨강 점선: SGD 검증
plt.plot(hist_adam.history['accuracy'],'g')         # 초록 실선: Adam 훈련
plt.plot(hist_adam.history['val_accuracy'],'g--')   # 초록 점선: Adam 검증
plt.plot(hist_adagrad.history['accuracy'],'b')      # 파랑 실선: Adagrad 훈련
plt.plot(hist_adagrad.history['val_accuracy'],'b--')# 파랑 점선: Adagrad 검증
plt.plot(hist_rmsprop.history['accuracy'],'m')      # 자홍 실선: RMSprop 훈련
plt.plot(hist_rmsprop.history['val_accuracy'],'m--')# 자홍 점선: RMSprop 검증
plt.title('Model accuracy comparison between optimizers')
plt.ylim((0.6,1.0))     # 차이를 크게 보려고 y축 범위를 0.6~1.0으로 좁힘
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train_sgd','Val_sgd','Train_adam','Val_adam','Train_adagrad','Val_adagrad','Train_rmsprop','Val_rmsprop'], loc='best')
plt.grid()
plt.show()
