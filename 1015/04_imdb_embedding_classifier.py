# 단어 임베딩과 완전연결층으로 IMDB 영화평의 긍정과 부정을 분류하는 예제

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
from keras.layers import Dense,Flatten,Embedding
from keras import preprocessing

# 자주 쓰이는 1만 단어만 사용한다. 드문 단어는 학습에 도움이 되지 않으면서 사전만 키운다
# 샘플 크기는 평 하나를 단어 512개로 맞춘다는 뜻이다
dic_siz=10000 # 사전의 크기(사전에 있는 단어 개수)
sample_siz=512 # 샘플의 크기

# 케라스가 제공하는 간소한 버전의 IMDB 읽기
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=dic_siz)
print(x_train.shape,x_test.shape)
print(x_train[0])

# 단어를 숫자, 숫자를 단어로 변환하는데 쓰는 표(표는 딕셔너리로 구현)
# 8-6의 원본과 달리 이 데이터는 이미 단어가 숫자로 바뀌어 있다
# 숫자를 다시 단어로 보려면 역방향 표가 필요하다
word2id=imdb.get_word_index()
id2word={word:id for id,word in word2id.items()}

for i in range(1,21):
    print(id2word[i],end='/')
    
embed_space_dim=16 # 16차원의 임베딩 공간

# 평의 길이가 저마다 다르므로 512로 통일한다
# 짧은 평은 앞에 0을 채우고 긴 평은 앞을 잘라낸다
x_train=preprocessing.sequence.pad_sequences(x_train,maxlen=sample_siz)
x_test=preprocessing.sequence.pad_sequences(x_test,maxlen=sample_siz)

# 신경망 모델 설계와 학습
embed=Sequential()
# Embedding은 단어 번호를 16차원 실수 벡터로 바꾸는 층이다
# 원핫 코드는 1만 차원이 필요하지만 임베딩은 16차원으로 줄이면서
# 뜻이 비슷한 단어가 가까이 모이도록 학습된다
# Flatten은 512개 단어의 16차원 벡터를 8192차원 벡터 하나로 펼친다
embed.add(Embedding(input_dim=dic_siz,output_dim=embed_space_dim,input_length=sample_siz))
embed.add(Flatten())
embed.add(Dense(32,activation='relu'))
embed.add(Dense(1,activation='sigmoid'))
embed.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
hist=embed.fit(x_train,y_train,epochs=20,batch_size=64,validation_data=(x_test,y_test),verbose=2)

embed.summary()

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