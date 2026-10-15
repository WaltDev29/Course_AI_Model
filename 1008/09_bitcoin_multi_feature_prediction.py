# 종가와 시가, 고가, 저가를 함께 입력해 네 값을 동시에 예측하는 예제

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
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
base_dir=os.path.dirname(os.path.abspath(__file__))
csv_path=os.path.join(base_dir,'BTC_USD.csv')

# 파일이 없으면 원인을 알기 어려운 오류가 나므로 미리 확인하고 안내한다
if not os.path.isfile(csv_path):
    print('가격 데이터를 찾을 수 없습니다:',csv_path)
    print('8-1.py를 먼저 실행해 BTC_USD.csv를 내려받은 뒤 다시 실행하세요.')
    sys.exit(1)

# 8-1이 저장한 csv에서 1년치 비트코인 가격 데이터 읽기
f=open(csv_path,'r')
coindesk_data=pd.read_csv(f,header=0)
# 8-2는 종가 하나만 썼고 여기서는 네 개를 쓴다
# 입력 특징이 늘면 LSTM 입력이 (7, 1)에서 (7, 4)로 바뀐다
seq=coindesk_data[['Closing Price (USD)','24h Open (USD)','24h High (USD)','24h Low (USD)']].to_numpy() # 종가, 시가, 고가, 저가를 모두 취함

# 시계열 데이터를 윈도우 단위로 자르는 함수
def seq2dataset(seq,window,horizon):
    X=[]; Y=[]
    for i in range(len(seq)-(window+horizon)+1):
        x=seq[i:(i+window)]
        y=(seq[i+window+horizon-1])
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

# 8-2와 같이 7일을 보고 다음 날을 맞힌다. 달라진 것은 하루에 쓰는 값의 개수뿐이다
w=7 # 윈도우 크기
h=1 # 수평선 계수

# X는 (샘플 수, 7, 4) 모양이 된다. 마지막 4가 종가, 시가, 고가, 저가다
# 8-2에서는 이 자리가 1이었다
X,Y = seq2dataset(seq,w,h)
print(X.shape,Y.shape)
print(X[0],Y[0])

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# 훈련 집합과 테스트 집합으로 분할
# 시계열이므로 섞지 않고 시간 순서대로 앞 70%를 훈련, 뒤 30%를 테스트로 쓴다
# 뒤쪽 날짜가 훈련에 들어가면 미래를 미리 본 셈이 되어 성능이 부풀려진다
split=int(len(X)*0.7)
x_train=X[0:split]; y_train=Y[0:split]
x_test=X[split:]; y_test=Y[split:]

# LSTM 모델의 설계와 학습
model = Sequential()
# input_shape가 (7, 4)로 자동으로 정해진다. 코드를 고칠 필요가 없다
model.add(LSTM(units=128,activation='relu',input_shape=x_train[0].shape))
# 출력도 네 개로 늘려 종가, 시가, 고가, 저가를 한꺼번에 예측한다
# 구조 변경은 input_shape와 이 Dense 뿐이고 나머지는 8-2와 같다
model.add(Dense(4))
# 값을 맞히는 회귀 문제이므로 교차 엔트로피가 아닌 평균절댓값오차를 쓴다
# mae는 달러 단위로 해석할 수 있어 직관적이다
model.compile(loss='mae',optimizer='adam',metrics=['mae'])
hist=model.fit(x_train,y_train,epochs=200,batch_size=1,validation_data=(x_test,y_test),verbose=2)

# LSTM 모델 평가
ev=model.evaluate(x_test,y_test,verbose=0)
print("손실 함수:",ev[0],"MAE:",ev[1])

# LSTM 모델로 예측 수행
# 예측도 네 개가 한꺼번에 나오므로 MAPE도 네 값이 출력된다
# 종가만 쓰는 8-2와 어느 쪽 오차가 작은지 비교해 보면 좋다
pred=model.predict(x_test) # LSTM
print("LSTM 평균절댓값백분율오차(MAPE):",sum(abs(y_test-pred)/y_test)/len(x_test))

# 학습 곡선
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('Model mae')
plt.ylabel('mae')
plt.xlabel('Epoch')
# 교재의 y축 범위 고정(plt.ylim([100,600]))은 2019~2020년 가격(수천 달러)에 맞춘 값이다
# 지금 데이터는 가격대가 20배 이상 높아 오차도 그만큼 커지므로 범위를 고정하지 않는다
plt.legend(['Train','Validation'], loc='best')
plt.grid()
plt.show()

# 예측 결과 시각화
# 8-3은 예측값이 네 개이므로 어느 것을 그릴지 골라야 한다
# 0번이 종가이고 1, 2, 3번이 각각 시가, 고가, 저가다. 숫자를 바꾸면 다른 값을 볼 수 있다
col=0
x_range=range(len(y_test))
plt.plot(x_range,y_test[x_range,col], color='red')
plt.plot(x_range,pred[x_range,col], color='blue')
plt.title('Closing price: true vs predicted')
plt.legend(['True prices','Predicted prices'], loc='best')
plt.grid()
plt.show()

# 일부 구간을 확대하여 시각화
# 전체 그래프에서는 잘 맞는 듯 보여도 확대하면 예측이 한 칸씩 늦게 따라가는 것이 보인다
# 직전 값을 거의 그대로 내놓는 현상이며 입력을 네 개로 늘려도 그대로 남는다
x_range=range(50,64)
plt.plot(x_range,y_test[x_range,col], color='red')
plt.plot(x_range,pred[x_range,col], color='blue')
plt.title('Closing price (zoom)')
plt.legend(['True prices','Predicted prices'], loc='best')
plt.grid()
plt.show()
