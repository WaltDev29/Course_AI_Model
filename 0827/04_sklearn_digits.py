# SVM으로 필기 숫자를 분류하고, 훈련 집합 자체로 정확률을 측정해보는 예제

from sklearn import datasets    # 예제 데이터셋 모음
from sklearn import svm         # SVM(서포트 벡터 머신) 분류/회귀 모델

digit=datasets.load_digits()    # 8x8 필기 숫자 영상 1,797장을 읽는다

# svm의 분류기 모델 SC를 학습
#   gamma : RBF 커널의 폭. 값이 클수록 결정 경계가 복잡해져 과잉적합 위험이 커진다
#   C     : 오분류 허용 정도. 값이 클수록 훈련 오류를 줄이는 쪽으로 강하게 맞춘다
s=svm.SVC(gamma=0.1,C=10)
s.fit(digit.data,digit.target) # digit 데이터로 모델링
# fit(특징 행렬, 정답 레이블) 형태로 호출한다. data는 64개 화솟값, target은 0~9 정답이다.

# 훈련 집합의 앞에 있는 샘플 3개를 새로운 샘플로 간주하고 인식해봄
new_d=[digit.data[0],digit.data[1],digit.data[2]]   # predict는 여러 샘플을 한꺼번에 받으므로 리스트로 묶는다
res=s.predict(new_d)                                # 세 샘플의 예측 레이블을 배열로 돌려준다
print("예측값은", res)
print("참값은", digit.target[0],digit.target[1],digit.target[2])

# 훈련 집합을 테스트 집합으로 간주하여 인식해보고 정확률을 측정
res=s.predict(digit.data)
correct=[i for i in range(len(res)) if res[i]==digit.target[i]]   # 예측과 정답이 일치하는 샘플의 번호만 모은다
accuracy=len(correct)/len(res)                                    # 정확률 = 맞힌 개수 / 전체 개수
print("화소 특징을 사용했을 때 정확률=",accuracy*100, "%")
# 주의: 학습에 쓴 데이터를 그대로 테스트에 썼으므로 이 값은 실제 성능보다 크게 부풀려진다.
#       올바른 평가는 훈련/테스트 집합을 나누는 3-5.py 방식을 따라야 한다.
