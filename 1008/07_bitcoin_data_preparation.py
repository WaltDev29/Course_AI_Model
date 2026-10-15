# 비트코인 가격 시계열을 읽어 LSTM 학습용 윈도우로 자르는 예제
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################################################################
# 야후 파이낸스에서 비트코인 가격을 직접 내려받아 csv로 저장한다
# 기존 코인데스크 파일과 열 이름을 똑같이 맞추므로 아래 코드는 그대로 쓸 수 있다
from pathlib import Path
import yfinance as yf

start_date='2025-10-01'
end_date='2026-10-01'

# 실행 위치와 상관없이 이 스크립트와 같은 폴더에 저장한다
BASE_DIR = Path(__file__).resolve().parents[1] / "data"
csv_path= BASE_DIR / 'BTC_USD.csv'

if not Path.is_file(csv_path): # 이미 받아 두었으면 다시 내려받지 않는다
    df=yf.download("BTC-USD",start=start_date,end=end_date,progress=False,auto_adjust=True)
    # yfinance는 ('Close','BTC-USD')처럼 2단 열 이름을 돌려주므로 한 단으로 낮춘다
    # 이 과정을 생략하면 csv 머리글이 세 줄로 저장되어 read_csv가 열을 찾지 못한다
    df.columns=df.columns.get_level_values(0)
    # 코인데스크 csv와 같은 열 이름으로 바꾼다
    df=df.rename(columns={'Close':'Closing Price (USD)','Open':'24h Open (USD)',
                          'High':'24h High (USD)','Low':'24h Low (USD)'})
    df.insert(0,'Currency','BTC')
    df=df[['Currency','Closing Price (USD)','24h Open (USD)','24h High (USD)','24h Low (USD)']]
    df.to_csv(csv_path)
    print('내려받기 완료:',csv_path,'/',len(df),'일치')
else:
    print('이미 있는 파일을 사용:',csv_path)
#################################################################

# 코인데스크 사이트에서 다운로드한 1년치 비트코인 가격 데이터 읽기
# csv에는 날짜, 종가, 시가, 고가, 저가가 들어 있다
# header=0은 첫 줄을 열 이름으로 쓰라는 뜻이다
f=open(csv_path,'r')
coindesk_data=pd.read_csv(f,header=0)
seq=coindesk_data[['Closing Price (USD)']].to_numpy() # 종가만 취함
print('데이터 길이:',len(seq),'\n앞쪽 5개 값:',seq[0:5])

# 그래프로 데이터 확인
# 학습 전에 데이터의 모양을 먼저 확인한다
# 추세와 변동 폭을 보고 윈도우 크기를 가늠할 수 있다
# x축을 0부터 세는 일련번호 대신 실제 날짜로 표시한다
dates=pd.to_datetime(coindesk_data['Date'])
plt.plot(dates,seq,color='red')
plt.title('Bitcoin Prices ('+start_date+' ~ '+end_date+')')
plt.xlabel('Date');plt.ylabel('Price in USD')
plt.xticks(rotation=45) # 날짜가 길어서 겹치므로 비스듬히 눕힌다
plt.tight_layout()      # 눕힌 글자가 그림 밖으로 잘리지 않게 여백을 맞춘다
plt.show()

# 시계열 데이터를 윈도우 단위로 자르는 함수
# 시계열 예측은 '과거 window일을 보고 horizon일 뒤를 맞히는' 문제로 바꾸어 푼다
# 한 칸씩 밀면서 자르므로 데이터 376개에서 샘플 370개가 나온다
# window=7, horizon=1이면 0~6일로 7일을 맞히고, 1~7일로 8일을 맞히는 식이다
def seq2dataset(seq,window,horizon):
    X=[]; Y=[]
    for i in range(len(seq)-(window+horizon)+1):
        x=seq[i:(i+window)]
        y=(seq[i+window+horizon-1])
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

# 윈도우를 키우면 더 긴 과거를 보지만 샘플 수가 줄어든다
# 수평선 계수는 몇 일 뒤를 예측할지를 정한다. 1이면 바로 다음 날이다
w=7 # 윈도우 크기
h=1 # 수평선 계수

# X는 (샘플 수, 7, 1), Y는 (샘플 수, 1) 모양이 된다
# LSTM은 (시간 길이, 특징 수) 형태의 입력을 기대한다
X,Y = seq2dataset(seq,w,h)
# 만들어진 학습 데이터의 모양을 확인한다
print('X 모양:',X.shape,'= (샘플 수, 보여줄 날짜 수, 하루당 값 개수)')
print('Y 모양:',Y.shape,'= (샘플 수, 맞힐 값 개수)')

# 첫 샘플과 마지막 샘플을 직접 확인한다
# X의 마지막 날 바로 다음 날이 Y이면 제대로 잘린 것이다
# 모양만 맞고 Y를 한 칸 잘못 잡으면 정답이 문제 안에 섞여 들어가는데 이 출력으로 걸러낸다
print('[첫 샘플]  입력 X[0] =',X[0].ravel())
print('           정답 Y[0] =',Y[0])
print('[끝 샘플] 입력 X[-1] =',X[-1].ravel())
print('          정답 Y[-1] =',Y[-1])
