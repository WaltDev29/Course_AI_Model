# 퍼셉트론으로 OR 게이트를 학습하는 예제 (sklearn Perceptron의 가장 기본적인 사용법)

from sklearn.linear_model import Perceptron

# 훈련 집합 구축
X=[[0,0],[0,1],[1,0],[1,1]]     # 특징 벡터 4개 (2차원 입력이므로 특징이 2개)
y=[-1,1,1,1]                    # 부류 레이블 (OR 게이트이므로 (0,0)만 -1이고 나머지는 +1)

# fit 함수로 Perceptron 학습
p=Perceptron()                  # 퍼셉트론 객체 생성 (하이퍼 매개변수는 모두 기본값 사용)
p.fit(X,y)                      # 훈련 집합을 보고 가중치와 편향을 스스로 찾아냄

# coef_는 가중치 [[w1,w2]], intercept_는 편향 [b] 이다.
# 속성 이름 뒤의 밑줄(_)은 fit 이후에 생기는 학습 결과라는 sklearn의 관례이다.
# 퍼셉트론의 예측식은 z=w1*x1+w2*x2+b 이고, z>0이면 +1, 아니면 -1로 분류한다.
print("학습된 퍼셉트론의 매개변수: ",p.coef_,p.intercept_)
print("훈련집합에 대한 예측: ",p.predict(X))        # 4개 샘플을 실제로 예측해 본 결과
print("정확률 측정: ",p.score(X,y)*100,"%")         # 맞춘 비율 (OR은 선형 분리가 되므로 100%)
