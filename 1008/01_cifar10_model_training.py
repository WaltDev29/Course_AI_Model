# CIFAR-10 CNN을 학습하고 배포용 모델 파일로 저장하는 예제

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

# 6장에서 쓰던 절차와 같다. 화솟값을 0~1로 정규화하고 레이블을 원핫 코드로 변환한다
# 배포할 때도 이와 똑같이 전처리해야 학습 때의 정확률이 재현된다
# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

# 6-4와 동일한 구조다. 이 장의 목적은 새 구조의 설계가 아니라
# 학습을 마친 모델을 파일로 저장하고 다른 프로그램에서 불러 쓰는 절차의 확인이다
# 신경망 모델 설계
cnn=Sequential()
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(512,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10,activation='softmax'))

# 에포크를 100으로 크게 잡았다. 배포용 모델이므로 정확률을 최대한 확보한다
# 실습 시간이 부족하면 30 정도로 낮춰도 이후 예제의 진행에는 지장이 없다
# 신경망 모델 학습
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
hist=cnn.fit(x_train,y_train,batch_size=128,epochs=100,validation_data=(x_test,y_test),verbose=2)

# 신경망 모델 정확률 평가
res=cnn.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)

import matplotlib.pyplot as plt

# 정확률 그래프
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='best')
plt.grid()
plt.show()

# 손실 함수 그래프
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='best')
plt.grid()
plt.show()

# 구조와 가중치를 파일 하나에 저장한다
# 7-2부터 7-5까지가 이 파일을 읽어 쓰므로 이 예제를 먼저 실행해야 한다
cnn.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"my_cnn_for_deploy.h5"))