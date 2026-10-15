# 세 곡을 함께 학습해 가락을 생성하는 예제(8-4의 데이터 확장)
# 8-4는 한 곡만 학습했고 여기서는 세 곡을 이어 붙여 학습한다
# 데이터가 늘면 한 곡을 외우는 대신 더 일반적인 가락을 배운다

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
os.environ["KERAS_BACKEND"]="torch"

# 지금 CPU로 도는지 GPU로 도는지 출력한다
# 위의 백엔드 지정이 적용되지 않으면 여기서 드러난다
import keras,torch
print("백엔드:",keras.backend.backend(),"/ 장치:","GPU "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import music21

# "작은 별", "봄나들이", "나비야 나비야" 악보를 ABC 표기로 표현
little_star="tinynotation: 4/4 c4 c4 g4 g4 a4 a4 g2 f4 f4 e4 e4 d4 d4 c2 g4 g4 f4 f4 e4 e4 d2 g4 g4 f4 f4 e4 e4 d2 c4 c4 g4 g4 a4 a4 g2 f4 f4 e4 e4 d4 d4 c2"
spring_picnic="tinynotation: 4/8 g8 e8 g8 e8 g8 a8 g4 e8 g8 e8 c8 d8 e8 c4 g8 e8 g8 e8 g8 a8 g4 b8 a8 g8 e8 d8 e8 c4"
butterfly="tinynotation: 2/4 g8 e8 e4 f8 d8 d4 c8 d8 e8 f8 g8 g8 g4 g8 e8 e8 e8 f8 d8 d4 c8 e8 g8 g8 e8 e8 e4 d8 d8 d8 d8 d8 e8 f4 e8 e8 e8 e8 e8 f8 g4 g8 e8 e4 f8 d8 d4 c8 e8 g8 g8 e8 e8 e4"

import numpy as np

# 계이름과 숫자를 상호 변환하는 표
note2num={'c':1,'d':2,'e':3,'f':4,'g':5,'a':6,'b':7}
num2note={1:'c',2:'d',3:'e',4:'f',5:'g',6:'a',7:'b'}

# ABC 표기를 시계열 데이터로 변환
def abc2timeseries(s):
    notes=s.split(' ')[2:]
    seq=[]
    for i in notes:
        seq.append([note2num[i[0]],int(i[1])])
    return seq

# 시계열 데이터를 ABC 표기로 변환
def timeseries2abc(t):
    s='tinynotation: 4/4'
    for i in t:
        s=s+' '+num2note[i[0]]+str(i[1])
    return s

# 원핫 코드로 변환하는 표
onehot=[[1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[1,4],[2,4],[3,4],[4,4],[5,4],[6,4],[7,4],[1,8],[2,8],[3,8],[4,8],[5,8],[6,8],[7,8]]

# 레이블을 원핫 코드로 변환
def to_onehot(l):
    t=[]
    for i in range(len(l)):
        a=np.zeros(len(onehot))
        a[onehot.index(list(l[i]))]=1.0
        t.append(a)
    return np.array(t)

# 시계열 데이터를 훈련 집합으로 자름
def seq2dataset(seq,window,horizon):
    X=[]; Y=[]
    for i in range(len(seq)-(window+horizon)+1):
        x=seq[i:(i+window)]
        y=(seq[i+window+horizon-1])
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

w=8 # 윈도우 크기
h=1 # 수평선 계수

# 세 곡을 시계열로 변환하고 결합
# 세 곡을 각각 시계열로 바꾼 뒤 하나로 이어 붙인다
# 이음매에서는 앞 곡의 끝과 뒤 곡의 시작이 한 윈도우에 섞여 들어간다
# 곡 수가 적을 때 쓰는 간단한 방법이며 곡별로 나누어 학습하는 방식도 있다
seq1=abc2timeseries(little_star)
seq2=abc2timeseries(butterfly)
seq3=abc2timeseries(spring_picnic)
seq=seq1+seq2+seq3

# 8-4는 한 곡이라 샘플이 적었고 세 곡을 이으면 샘플 수가 크게 늘어난다
X,Y = seq2dataset(seq,w,h)

from keras.models import Sequential
from keras.layers import Dense, LSTM

# 훈련 집합 구축
# 생성이 목적이라 맞는지 틀렸는지 점수를 매길 정답이 없다
# 그래서 테스트 집합을 두지 않고 전부 학습에 쓴다
split=int(len(X)*1.0) # 100%를 훈련집합으로 사용
x_train=X[0:split]; y_train=Y[0:split]
y_train=to_onehot(y_train)

# LSTM 모델 설계와 학습
model=Sequential()
# 신경망 구조는 8-4와 완전히 같다. 달라지는 것은 학습 데이터의 양뿐이다
model.add(LSTM(units=128,activation='relu',input_shape=x_train[0].shape))
# 출력 수는 원핫 표의 길이인 21로 자동으로 정해진다
# 계이름 7가지와 음 길이 3가지를 조합한 21부류 분류 문제다
model.add(Dense(y_train.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=200,batch_size=1,verbose=2)

# 학습된 모델로 편곡을 하는 함수(first_measure: 첫 소절, duration: 생성될 곡의 길이)
def arranging_music(model,first_measure,duration):
    music=first_measure
    for i in range(duration):
        p=model.predict(np.float32(np.expand_dims(music[-w:],axis=0)))
        music=np.append(music,[onehot[np.argmax(p)]],axis=0)
    return timeseries2abc(music)

# 첫 소절 8개 음을 주고 이어서 50개 음을 생성하게 한다
# 데이터가 늘어난 덕에 8-4보다 세 곡의 색이 섞인 가락이 나온다
new_song=arranging_music(model,x_train[0],50)

print(new_song)
music21.converter.parse(new_song).show('mid')