# 교차 검증으로 MLP의 은닉 노드 개수(하이퍼 매개변수)를 최적화하고 검증 곡선을 그리는 예제

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,validation_curve
import numpy as np
import matplotlib.pyplot as plt
import time

# 데이터셋을 읽고 훈련 집합과 테스트 집합으로 분할
digit=datasets.load_digits()    # 8x8 흑백 숫자 영상 1797장 (data는 64차원 특징, target은 0~9 정답)
x_train,x_test,y_train,y_test=train_test_split(digit.data,digit.target,train_size=0.6)  # 훈련 60%, 테스트 40%로 무작위 분할

# 다층 퍼셉트론을 교차 검증으로 성능 평가 (소요 시간 측정 포함)
start=time.time() # 시작 시각
mlp=MLPClassifier(learning_rate_init=0.001,batch_size=32,max_iter=300,solver='sgd')  # 은닉 노드 개수는 아래에서 바꿔 가며 실험하므로 여기서는 지정하지 않음
prange=range(50,1001,50)        # 실험할 은닉 노드 개수 후보 (50, 100, 150, ..., 1000으로 20가지)
# validation_curve는 param_range의 값마다 모델을 cv겹 교차 검증하여 훈련/검증 점수를 돌려준다.
# cv=10이므로 후보 하나당 10번 학습하고, n_jobs=4는 CPU 코어 4개로 병렬 처리하라는 뜻이다.
train_score,test_score=validation_curve(mlp,x_train,y_train,param_name="hidden_layer_sizes",param_range=prange,cv=10,scoring="accuracy",n_jobs=4)
end=time.time() # 끝난 시각
print("하이퍼 매개변수 최적화에 걸린 시간은",end-start,"초입니다.")

# 교차 검증 결과의 평균과 분산 구하기
# train_score와 test_score는 (후보 개수 x cv) 모양이므로 axis=1로 10번의 실험을 요약한다.
train_mean = np.mean(train_score,axis=1)    # 후보별 훈련 정확률 평균
train_std = np.std(train_score,axis=1)      # 후보별 훈련 정확률 표준편차 (실험 간 편차)
test_mean = np.mean(test_score,axis=1)      # 후보별 검증 정확률 평균
test_std = np.std(test_score,axis=1)        # 후보별 검증 정확률 표준편차

# 성능 그래프 그리기
plt.plot(prange,train_mean,label="Train score",color="r")   # 훈련 정확률 곡선 (빨강)
plt.plot(prange,test_mean,label="Test score",color="b")     # 검증 정확률 곡선 (파랑)
plt.fill_between(prange,train_mean-train_std,train_mean+train_std,alpha=0.2,color="r")  # 평균 ±1 표준편차를 띠로 표시
plt.fill_between(prange,test_mean-test_std,test_mean+test_std,alpha=0.2,color="b")      # 띠가 넓을수록 결과가 불안정하다는 뜻이다.
plt.legend(loc="best")
plt.title("Validation Curve with MLP")
plt.xlabel("Number of hidden nodes"); plt.ylabel("Accuracy")
plt.ylim(0.9,1.01)              # 차이를 크게 보기 위해 y축 범위를 0.9~1.01로 좁힘
plt.grid(axis='both')
plt.show()

best_number_nodes=prange[np.argmax(test_mean)] # 최적의 은닉 노드 개수
print("\n최적의 은닉층의 노드 개수는",best_number_nodes,"개입니다.\n")

# 최적의 은닉 노드 개수로 모델링
# 위에서 찾은 최적값으로 훈련 집합 전체를 다시 학습한다.
mlp_test=MLPClassifier(hidden_layer_sizes=(best_number_nodes),learning_rate_init=0.001,batch_size=32,max_iter=300,solver='sgd')
mlp_test.fit(x_train,y_train)

# 테스트 집합으로 예측
res=mlp_test.predict(x_test)    # 하이퍼 매개변수 선택에 쓰지 않은 데이터로 최종 성능을 확인

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
print("테스트 집합에 대한 정확률은", accuracy*100, "%입니다.")
