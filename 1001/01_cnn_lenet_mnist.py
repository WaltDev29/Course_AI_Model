# LeNet-5 구조의 CNN으로 MNIST 손글씨 숫자를 분류하는 예제

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
os.environ["KERAS_BACKEND"]="torch"

# 지금 CPU로 도는지 GPU로 도는지 출력한다
# 위의 백엔드 지정이 적용되지 않으면 여기서 드러난다
import keras,torch
print("백엔드:",keras.backend.backend(),"/ 장치:","GPU "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.optimizers import Adam

# MNIST 데이터셋을 읽고 신경망에 입력할 형태로 변환
# 원래 (60000,28,28) 모양이므로 맨 뒤에 채널 축을 붙여 (60000,28,28,1)로 바꾼다
# 컨볼루션층은 항상 (높이,너비,채널) 3차원 입력을 기대하기 때문이다
(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
# 0~255 정수를 0~1 실수로 정규화해 학습을 안정시킨다
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
# 레이블 3을 [0,0,0,1,0,0,0,0,0,0]처럼 원핫 코드로 바꾼다(손실함수가 요구하는 형태)
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

# LeNet-5 신경망 모델 설계
# 컨볼루션층과 풀링층을 번갈아 쌓아 특징을 추출하고 완전연결층으로 분류한다
# padding='same'은 출력 크기를 입력과 같게 유지하라는 뜻이다
cnn=Sequential()
cnn.add(Conv2D(6,(5,5),padding='same',activation='relu',input_shape=(28,28,1)))
cnn.add(MaxPooling2D(pool_size=(2,2)))  # 28x28 -> 14x14로 축소
cnn.add(Conv2D(16,(5,5),padding='same',activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))  # 14x14 -> 7x7로 축소
cnn.add(Conv2D(120,(5,5),padding='same',activation='relu'))
cnn.add(Flatten())  # 3차원 특징 맵을 1차원 벡터로 펼쳐 완전연결층에 연결
cnn.add(Dense(84,activation='relu'))
cnn.add(Dense(10,activation='softmax'))  # 10개 부류의 확률을 출력

# 신경망 모델 학습
# 부류가 여러 개이고 레이블이 원핫이므로 categorical_crossentropy를 쓴다
# validation_data를 주면 에포크마다 테스트 집합 성능도 함께 기록된다
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
hist=cnn.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)

# 신경망 모델 정확률 평가
# evaluate는 [손실, 정확률]을 반환하므로 res[1]이 정확률이다
res=cnn.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)

import matplotlib.pyplot as plt

# 정확률 그래프
# 두 곡선이 함께 올라가면 정상이고, 훈련만 오르고 검증이 꺾이면 과잉적합이다
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

# 손실 함수 그래프
# 검증 손실이 다시 올라가기 시작하는 지점이 과잉적합이 시작되는 시점이다
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()
