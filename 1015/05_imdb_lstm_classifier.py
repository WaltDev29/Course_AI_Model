# 단어 임베딩과 LSTM으로 IMDB 영화평을 분류하는 예제(8-7과 비교)
# 8-7은 Flatten으로 단어를 한꺼번에 펼쳤고 여기서는 LSTM으로 순서대로 읽는다
# 단어의 순서를 쓰는지 여부가 두 예제의 차이다

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
os.environ["KERAS_BACKEND"]="torch"

# 지금 CPU로 도는지 GPU로 도는지 출력한다
# 위의 백엔드 지정이 적용되지 않으면 여기서 드러난다
import keras,torch
print("백엔드:",keras.backend.backend(),"/ 장치:","GPU "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
from keras import preprocessing
from keras.callbacks import EarlyStopping

# 자주 쓰이는 1만 단어만 쓰고 평 하나를 단어 512개로 맞춘다
# 8-7과 같은 값이라 두 모델의 차이를 구조 차이로만 볼 수 있다
dic_siz=10000 # 사전의 크기(사전에 있는 단어 개수)
sample_siz=512 # 샘플의 크기

# 케라스가 제공하는 간소한 버전의 IMDB 읽기
# 8-6의 원본과 달리 이 데이터는 단어가 이미 숫자로 바뀌어 있다
# 레이블도 긍정 1, 부정 0으로 정리되어 있다
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=dic_siz)

embed_space_dim=16 # 16차원의 임베딩 공간

# 평의 길이가 저마다 다르므로 512로 통일한다
# 짧은 평은 앞에 0을 채우고 긴 평은 앞을 잘라낸다
x_train=preprocessing.sequence.pad_sequences(x_train,maxlen=sample_siz)
x_test=preprocessing.sequence.pad_sequences(x_test,maxlen=sample_siz)

# 검증 정확률이 5 에포크 동안 나아지지 않으면 학습을 멈춘다
# restore_best_weights는 멈춘 시점이 아니라 가장 좋았던 시점의 가중치로 되돌린다
early=EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True)

# 신경망 모델의 설계와 학습(LSTM 층 포함)
embed=Sequential()
# Embedding은 단어 번호를 16차원 실수 벡터로 바꾸는 층이다
# 뜻이 비슷한 단어가 가까이 모이도록 학습된다
embed.add(Embedding(input_dim=dic_siz,output_dim=embed_space_dim,input_length=sample_siz))
# LSTM은 512개 단어를 앞에서 뒤로 읽으며 기억을 갱신하고 마지막 상태만 내놓는다
# Flatten을 쓴 8-7보다 매개변수가 훨씬 적으면서 어순을 반영한다
embed.add(LSTM(units=32))
embed.add(Dense(1,activation='sigmoid'))
embed.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
# 8-7과 달리 테스트 집합을 검증에 쓰지 않고 훈련 집합의 20%를 떼어 검증한다
# 조기 종료 판단에 테스트 집합을 쓰면 테스트 집합이 학습에 개입하게 된다
hist=embed.fit(x_train,y_train,epochs=20,batch_size=64,validation_split=0.2,verbose=2,callbacks=[early])

# 모델 평가
res=embed.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)

import matplotlib.pyplot as plt

# 학습 곡선
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='best')
plt.grid()
plt.show()