# CIFAR-10을 CNN으로 분류하고 학습이 끝난 모델을 파일로 저장하는 예제

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

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
# CIFAR-10은 32x32 컬러 영상이라 처음부터 (50000,32,32,3) 모양이다
# 앞 예제들과 달리 채널 축을 붙이는 reshape이 필요 없다
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

# 신경망 모델 설계
# 컬러 영상은 명암 영상보다 어려우므로 [Conv-Conv-Pool-Dropout] 블록을 두 번 쌓는다
# 뒤쪽 블록일수록 커널 수를 32에서 64로 늘려 더 복잡한 특징을 잡게 한다
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

# 신경망 모델 학습
# 30 에포크라 시간이 꽤 걸린다. 학습이 끝나면 아래에서 파일로 저장하므로
# 6-5부터는 다시 학습하지 않고 저장된 모델을 불러 쓰면 된다
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
hist=cnn.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)

# 신경망 모델 정확률 평가
res=cnn.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)

import matplotlib.pyplot as plt

# 정확률 그래프
# CIFAR-10은 과잉적합이 뚜렷해서 훈련 곡선과 검증 곡선의 간격이 크게 벌어진다
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
# 신경망 구조와 가중치를 하나의 파일에 저장(6-5가 이 파일을 읽어 씀)
# 실행 위치가 아니라 이 스크립트와 같은 폴더에 저장해야 6-5가 찾을 수 있다
cnn.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"my_cnn.h5"))
