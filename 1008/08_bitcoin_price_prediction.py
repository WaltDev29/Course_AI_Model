# LSTM으로 비트코인 종가의 다음 날 값을 예측하는 예제

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
from pathlib import Path
os.environ["KERAS_BACKEND"]="torch"

# 지금 CPU로 도는지 GPU로 도는지 출력한다
# 위의 백엔드 지정이 적용되지 않으면 여기서 드러난다
import keras,torch
print("백엔드:",keras.backend.backend(),"/ 장치:","GPU "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

# 8-1이 야후 파이낸스에서 내려받아 저장한 csv를 읽는다
# 실행 위치와 상관없이 이 스크립트와 같은 폴더를 기준으로 찾는다
BASE_DIR = Path(__file__).resolve().parents[1] / "data"
csv_path= BASE_DIR / 'BTC_USD.csv'

# 파일이 없으면 원인을 알기 어려운 오류가 나므로 미리 확인하고 안내한다
if not os.path.isfile(csv_path):
    print('가격 데이터를 찾을 수 없습니다:',csv_path)
    print('8-1.py를 먼저 실행해 BTC_USD.csv를 내려받은 뒤 다시 실행하세요.')
    sys.exit(1)

# 8-1이 저장한 csv에서 1년치 비트코인 가격 데이터 읽기
f=open(csv_path,'r')
coindesk_data=pd.read_csv(f,header=0)
seq=coindesk_data[['Closing Price (USD)']].to_numpy() # 종가만 취함

# 시계열 데이터를 윈도우 단위로 자르는 함수
def seq2dataset(seq,window,horizon):
    X=[]; Y=[]
    for i in range(len(seq)-(window+horizon)+1):
        x=seq[i:(i+window)]
        y=(seq[i+window+horizon-1])
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

w=7 # 윈도우 크기
h=1 # 수평선 계수

X,Y=seq2dataset(seq,w,h)

# 데이터 준비가 끝난 뒤에 신경망 관련 모듈을 가져온다
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# 훈련 집합과 테스트 집합으로 분할
# 시계열은 무작위로 섞어 나누면 안 된다
# 뒤쪽 날짜가 훈련 집합에 들어가면 미래를 미리 본 셈이 되어 성능이 부풀려진다
# 그래서 앞 70%를 훈련, 뒤 30%를 테스트로 시간 순서대로 자른다
split=int(len(X)*0.7)
x_train=X[0:split]; y_train=Y[0:split]
x_test=X[split:]; y_test=Y[split:]

# LSTM 모델 설계와 학습
model=Sequential()
# LSTM은 순서가 있는 데이터를 앞에서 뒤로 읽으며 기억을 갱신한다
# units=128은 기억 상태의 크기이고 input_shape는 (7, 1)이 된다
# Dense(1)은 다음 날 종가 하나를 실수로 출력한다
model.add(LSTM(units=128,activation='relu',input_shape=x_train[0].shape))
model.add(Dense(1))
# 분류가 아니라 값을 맞히는 회귀 문제이므로 교차 엔트로피가 아닌 오차를 쓴다
# mae는 평균절댓값오차이며 달러 단위로 해석할 수 있어 직관적이다
model.compile(loss='mae',optimizer='adam',metrics=['mae'])
hist=model.fit(x_train,y_train,epochs=200,batch_size=1,validation_data=(x_test,y_test),verbose=2)

# LSTM 모델 평가
ev=model.evaluate(x_test,y_test,verbose=0)
print("손실 함수:",ev[0],"MAE:",ev[1])

# LSTM 모델로 예측 수행
# MAPE는 오차를 실제 가격으로 나눈 비율이다
# 가격 수준이 달라도 비교할 수 있어 시계열 예측에서 많이 쓴다
pred=model.predict(x_test)
print("평균절댓값백분율오차(MAPE):",sum(abs(y_test-pred)/y_test)/len(x_test))

# 학습 곡선
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('Model mae')
plt.ylabel('mae')
plt.xlabel('Epoch')
# 교재의 y축 범위 고정(plt.ylim([120,800]))은 2019~2020년 가격(수천 달러)에 맞춘 값이다
# 지금 데이터는 가격대가 20배 이상 높아 오차도 그만큼 커지므로 범위를 고정하지 않는다
plt.legend(['Train','Validation'], loc='best')
plt.grid()
plt.show()

# 예측 결과 시각화
x_range=range(len(y_test))
plt.plot(x_range,y_test[x_range], color='red')
plt.plot(x_range,pred[x_range], color='blue')
plt.legend(['True prices','Predicted prices'], loc='best')
plt.grid()
plt.show()

# 일부 구간을 확대하여 시각화
# 전체 그래프에서는 잘 맞는 듯 보여도 확대하면 한 칸씩 늦게 따라가는 것이 보인다
# 직전 값을 거의 그대로 내놓는 현상이며 시계열 예측의 대표적 함정이다
x_range=range(50,64)
plt.plot(x_range,y_test[x_range], color='red')
plt.plot(x_range,pred[x_range], color='blue')
plt.legend(['True prices','Predicted prices'], loc='best')
plt.grid()
plt.show()