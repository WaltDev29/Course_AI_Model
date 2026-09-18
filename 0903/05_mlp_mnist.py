# 다층 퍼셉트론(MLP)으로 MNIST 손글씨 숫자 7만 장을 분류하는 예제 (digits보다 큰 실전 데이터셋)

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import numpy as np

# MNIST 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
mnist=fetch_openml('mnist_784')                     # OpenML에서 내려받음 (28x28=784차원 특징, 7만 장)
mnist.data=mnist.data/255.0                         # 화솟값 0~255를 0~1로 정규화 (학습 안정에 필요)
x_train=mnist.data[:60000]; x_test=mnist.data[60000:]                   # 앞 6만 장은 훈련, 뒤 1만 장은 테스트 (MNIST의 표준 분할)
y_train=np.int16(mnist.target[:60000]); y_test=np.int16(mnist.target[60000:])   # 레이블이 문자열이므로 정수로 변환

# MLP 분류기 모델을 학습
# 은닉 노드 100개, 미니배치 512, 최대 300세대이며, solver='adam'은 학습률을 자동 조절하는 최적화 기법이다.
mlp=MLPClassifier(hidden_layer_sizes=(100),learning_rate_init=0.001,batch_size=512,max_iter=300,solver='adam',verbose=True)
mlp.fit(x_train,y_train)                            # 데이터가 크므로 학습에 수 분이 걸릴 수 있다.

# 테스트 집합으로 예측
res=mlp.predict(x_test)

# 혼동 행렬
# 행은 모델이 예측한 부류, 열은 실제 정답 부류이다. 대각선이 맞게 분류한 개수이다.
conf=np.zeros((10,10),dtype=np.int16)   # 개수를 세는 용도이므로 정수형으로 생성
for i in range(len(res)):
    conf[res[i]][y_test[i]]+=1          # (예측값, 정답) 칸의 개수를 1 증가
print(conf)

# 정확률 계산
no_correct=0
for i in range(10):
    no_correct+=conf[i][i]              # 대각선 원소의 합이 곧 맞게 분류한 샘플 수이다.
accuracy=no_correct/len(res)            # 맞춘 개수를 전체 테스트 샘플 수로 나눔
print("테스트 집합에 대한 정확률은", accuracy*100, "%입니다.")
