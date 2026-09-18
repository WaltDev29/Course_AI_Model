# 퍼셉트론(단층)으로 sklearn digits 손글씨 숫자를 분류하고 혼동 행렬로 성능을 평가하는 예제

from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
digit=datasets.load_digits()    # 8x8 흑백 숫자 영상 1797장 (data는 64차원 특징, target은 0~9 정답)
x_train,x_test,y_train,y_test=train_test_split(digit.data,digit.target,train_size=0.6)  # 훈련 60%, 테스트 40%로 무작위 분할

# fit 함수로 Perceptron 학습
# max_iter는 최대 세대(epoch) 수, eta0는 학습률, verbose=0은 학습 과정을 출력하지 않는다는 뜻이다.
p=Perceptron(max_iter=100,eta0=0.001,verbose=1)
p.fit(x_train,y_train) # digit 데이터로 모델링

res=p.predict(x_test) # 테스트 집합으로 예측

# 혼동 행렬
# 행은 모델이 예측한 부류, 열은 실제 정답 부류이다. 대각선이 맞게 분류한 개수이다.
conf=np.zeros((10,10))          # 부류가 0~9로 10개이므로 10x10 행렬을 0으로 초기화
for i in range(len(res)):
    conf[res[i]][y_test[i]]+=1  # (예측값, 정답) 칸의 개수를 1 증가
print(conf)

# 정확률 계산
no_correct=0
for i in range(10):
    no_correct+=conf[i][i]      # 대각선 원소의 합이 곧 맞게 분류한 샘플 수이다.
accuracy=no_correct/len(res)    # 맞춘 개수를 전체 테스트 샘플 수로 나눔
print("테스트 집합에 대한 정확률은 ", accuracy*100, "%입니다.")
