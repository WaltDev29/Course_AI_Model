# 5-7과 똑같은 MLP 구조로 fashion MNIST(의류 영상)를 분류하는 예제 (데이터셋만 바뀜)

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# fashion MNIST 데이터셋을 읽어와 신경망에 입력할 형태로 변환
# 크기와 개수는 MNIST와 같지만 숫자 대신 티셔츠, 바지, 가방 등 의류 10종을 담고 있다.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# 완전연결층은 1차원 벡터만 받으므로 28x28 영상을 784개의 값으로 펼친다.
x_train = x_train.reshape(60000,784) # 텐서 모양 변환
x_test = x_test.reshape(10000,784)
# 0~255 정수를 실수로 바꾸고 255로 나눠 0~1 범위로 정규화한다.
x_train=x_train.astype(np.float32)/255.0 # ndarray로 변환
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10) # 원핫 코드로 변환
y_test=tf.keras.utils.to_categorical(y_test,10)

n_input=784     # 입력 노드 개수 (28x28을 펼친 값)
n_hidden=1024   # 은닉 노드 개수
n_output=10     # 출력 노드 개수 (의류 10종)

mlp=Sequential()
# 5-7과 완전히 동일한 구조이다. 같은 모델이 데이터셋에 따라 성능이 얼마나 달라지는지 보는 것이 이 예제의 목적이다.
mlp.add (Dense(units=n_hidden,activation='tanh',input_shape=(n_input,),kernel_initializer='random_uniform',bias_initializer='zeros'))
mlp.add(Dense(units=n_output,activation='tanh',kernel_initializer='random_uniform',bias_initializer='zeros'))

mlp.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
hist=mlp.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)

res=mlp.evaluate(x_test,y_test,verbose=0)
# MNIST보다 정확률이 뚜렷하게 낮게 나온다. 의류 영상이 손글씨 숫자보다 구별하기 어렵기 때문이다.
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
# 5-7의 MNIST 그래프와 견주어 보면 두 곡선의 벌어지는 정도가 확연히 다르다.
plt.plot(hist.history['loss'])          # 훈련 집합 손실
plt.plot(hist.history['val_loss'])      # 검증(테스트) 집합 손실
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper right')
plt.grid()
plt.show()
