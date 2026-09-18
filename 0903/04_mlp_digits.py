# 다층 퍼셉트론(MLP)으로 digits를 분류하는 예제 (4-2의 단층 퍼셉트론과 성능을 비교하는 용도)

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
digit=datasets.load_digits()    # 8x8 흑백 숫자 영상 1797장 (data는 64차원 특징, target은 0~9 정답)
x_train,x_test,y_train,y_test=train_test_split(digit.data,digit.target,train_size=0.6)  # 훈련 60%, 테스트 40%로 무작위 분할

# MLP 분류기 모델을 학습
# hidden_layer_sizes는 은닉층 구조(노드 100개짜리 은닉층 1개), learning_rate_init은 초기 학습률,
# batch_size는 미니배치 크기, max_iter는 최대 세대 수, solver는 최적화 기법(sgd는 확률적 경사 하강법),
# verbose=True는 세대마다 손실값을 출력하라는 뜻이다.
mlp=MLPClassifier(hidden_layer_sizes=(100),learning_rate_init=0.001,batch_size=32,max_iter=300,solver='sgd',verbose=True)
mlp.fit(x_train,y_train)        # 오류 역전파 알고리즘으로 가중치를 학습

res=mlp.predict(x_test) # 테스트 집합으로 예측

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
