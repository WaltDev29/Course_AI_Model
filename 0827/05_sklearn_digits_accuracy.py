# 훈련/테스트 집합을 나누어 SVM을 학습하고, 혼동 행렬로 성능을 분석하는 예제

from sklearn import datasets                          # 예제 데이터셋 모음
from sklearn import svm                               # SVM 분류 모델
from sklearn.model_selection import train_test_split  # 데이터를 훈련용과 테스트용으로 나누는 함수
import numpy as np

# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
#   train_size=0.6 : 60%를 학습에, 나머지 40%를 평가에 사용
#   기본으로 무작위 분할이므로 실행할 때마다 정확률이 조금씩 달라진다.
#   결과를 고정하려면 random_state=숫자를 함께 지정한다.
digit=datasets.load_digits()
x_train,x_test,y_train,y_test=train_test_split(digit.data,digit.target,train_size=0.6)

# svm의 분류 모델 SVC를 학습
s=svm.SVC(gamma=0.001)   # gamma가 작을수록 결정 경계가 완만해져 일반화에 유리한 편이다
s.fit(x_train,y_train)   # 훈련 집합만으로 학습한다. 테스트 집합은 절대 학습에 쓰지 않는다

res=s.predict(x_test)    # 학습에 쓰지 않은 테스트 집합을 예측한다

# 혼동 행렬 구함
# conf[예측값][참값] 형태로 개수를 센다. 즉 행이 예측, 열이 정답이다.
# 대각선(i==i)은 맞힌 경우, 대각선 밖은 틀린 경우이며
# 어떤 숫자를 어떤 숫자로 착각했는지 한눈에 볼 수 있다.
conf=np.zeros((10,10))   # 숫자가 0~9이므로 10x10 행렬을 0으로 초기화
for i in range(len(res)):
    conf[res[i]][y_test[i]]+=1
print(conf)

# 정확률 측정하고 출력
no_correct=0
for i in range(10):
    no_correct+=conf[i][i]    # 대각선 값의 합 = 정확히 맞힌 샘플 수
accuracy=no_correct/len(res)  # 정확률 = 맞힌 개수 / 테스트 샘플 수
print("테스트 집합에 대한 정확률은", accuracy*100, "%입니다.")
# 3-4.py와 달리 학습에 쓰지 않은 데이터로 평가했으므로, 이 값이 실제 성능에 가깝다.
