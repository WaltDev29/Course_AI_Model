# LSTM으로 동요 한 곡을 학습해 이어지는 가락을 생성하는 예제
# music21은 악보를 다루는 라이브러리다
# show('mid')로 연주를 들으려면 MIDI 재생 프로그램이 함께 설정되어 있어야 한다

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
os.environ["KERAS_BACKEND"]="torch"

# 지금 CPU로 도는지 GPU로 도는지 출력한다
# 위의 백엔드 지정이 적용되지 않으면 여기서 드러난다
import keras,torch
print("백엔드:",keras.backend.backend(),"/ 장치:","GPU "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import music21

# "작은 별" 악보를 ABC 표기로 표현
little_star="tinynotation: 4/4 c4 c4 g4 g4 a4 a4 g2 f4 f4 e4 e4 d4 d4 c2 g4 g4 f4 f4 e4 e4 d2 g4 g4 f4 f4 e4 e4 d2 c4 c4 g4 g4 a4 a4 g2 f4 f4 e4 e4 d4 d4 c2"
music21.converter.parse(little_star).show('mid') # 스피커로 연주를 들려줌

import numpy as np

# 계이름과 숫자를 상호 변환하는 표(딕셔너리 자료구조를 사용함)
# 신경망은 글자를 다루지 못하므로 계이름을 숫자로 바꾼다
# 'c4'는 [1, 4]가 되는데 앞은 계이름이고 뒤는 음의 길이다
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
# 계이름 7가지와 길이 3가지를 조합한 21가지를 하나의 부류로 본다
# 즉 다음 음을 맞히는 21부류 분류 문제로 바꾸어 푼다
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

# 앞의 음 8개를 보고 다음 음 1개를 맞힌다
w=8 # 윈도우 크기
h=1 # 수평선 계수

seq=abc2timeseries(little_star)
X,Y=seq2dataset(seq,w,h)
print(X.shape,Y.shape)
print(X[0],Y[0])

from keras.models import Sequential
from keras.layers import Dense, LSTM

# 훈련 집합 구축
split=int(len(X)*1.0) # 100%를 훈련집합으로 사용
x_train=X[0:split]; y_train=Y[0:split]
y_train=to_onehot(y_train)

# LSTM 모델 설계와 학습
model=Sequential()
model.add(LSTM(units=128,activation='relu',input_shape=x_train[0].shape))
# 8-2와 달리 값을 맞히는 회귀가 아니라 21가지 중 하나를 고르는 분류다
# 그래서 softmax와 categorical_crossentropy를 쓴다
model.add(Dense(y_train.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=200,batch_size=1,verbose=2)

# 학습된 모델로 편곡을 하는 함수(first_measure: 첫 소절, duration: 생성될 곡의 길이)
# 예측한 음을 다시 입력 끝에 붙여 그다음 음을 예측하는 방식을 반복한다
# music[-w:]는 가장 최근 8개 음만 잘라 쓴다는 뜻이다
# 스스로 만든 음을 입력으로 되먹이므로 틀리기 시작하면 오류가 누적된다
def arranging_music(model,first_measure,duration):
    music=first_measure
    for i in range(duration):
        p=model.predict(np.float32(np.expand_dims(music[-w:],axis=0)))
        music=np.append(music,[onehot[np.argmax(p)]],axis=0)
    return timeseries2abc(music)

#new_song=arranging_music(model,x_train[0],50)
new_song=arranging_music(model,[[5,4],[3,4],[3,2],[4,4],[2,4],[2,2],[1,4],[2,4]],50)
 
print(new_song)
music21.converter.parse(new_song).show('mid')