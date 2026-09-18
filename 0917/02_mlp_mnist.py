# MNIST를 은닉층 1개짜리 다층 퍼셉트론(MLP)으로 분류하고 학습 곡선을 그리는 예제

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
# 0~255 정수를 실수로 바꾸고 255로 나눠 0~1 범위로 정규화한다. 학습 안정에 꼭 필요한 단계이다.
x_train=x_train.astype(np.float32)/255.0 # ndarray로 변환
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10) # 원핫 코드로 변환
y_test=tf.keras.utils.to_categorical(y_test,10)

n_input=784     # 입력 노드 개수 (28x28을 펼친 값)
n_hidden=1024   # 은닉 노드 개수
n_output=10     # 출력 노드 개수 (0~9 부류가 10개)

mlp=Sequential()
# 은닉층. 784개 입력을 받아 1024개 노드로 연결한다.
mlp.add (Dense(units=n_hidden,activation='tanh',input_shape=(n_input,),kernel_initializer='random_uniform',bias_initializer='zeros'))
# 출력층. 부류가 10개이므로 노드도 10개이다.
mlp.add(Dense(units=n_output,activation='tanh',kernel_initializer='random_uniform',bias_initializer='zeros'))

# Adam은 학습률을 자동으로 조절해 주는 옵티마이저이다. metrics에 accuracy를 넣어 정확률을 함께 지켜본다.
mlp.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
# validation_data를 주면 세대마다 테스트 집합의 성능도 함께 기록한다.
# 반환값 hist에는 세대별 손실과 정확률이 저장되어 아래에서 그래프로 그릴 수 있다.
hist=mlp.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)

res=mlp.evaluate(x_test,y_test,verbose=0)   # [손실, 정확률] 형태로 반환된다.
print("정확률은",res[1]*100)
import matplotlib.pyplot as plt

# 정확률 곡선
# 두 곡선이 함께 올라가면 정상이고, 훈련만 오르고 검증이 정체되면 과적합 신호이다.
plt.plot(hist.history['accuracy'])      # 훈련 집합 정확률
plt.plot(hist.history['val_accuracy'])  # 검증(테스트) 집합 정확률
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper left')
plt.grid()
plt.show()

# 손실 함수 곡선
# 검증 손실이 어느 시점부터 다시 올라가면 그 지점이 과적합이 시작되는 곳이다.
plt.plot(hist.history['loss'])          # 훈련 집합 손실
plt.plot(hist.history['val_loss'])      # 검증(테스트) 집합 손실
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper right')
plt.grid()
plt.show()
