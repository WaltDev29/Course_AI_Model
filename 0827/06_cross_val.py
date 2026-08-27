# 교차 검증(cross validation)으로 SVM 분류기의 성능을 안정적으로 평가하는 예제

from sklearn import datasets                          # 예제 데이터셋 모음
from sklearn import svm                               # SVM 분류 모델
from sklearn.model_selection import cross_val_score   # 교차 검증을 한 줄로 수행하는 함수

digit=datasets.load_digits()      # 8x8 필기 숫자 영상 1,797장
s=svm.SVC(gamma=0.001)            # 아직 학습하지 않은 모델. cross_val_score가 내부에서 복제해 학습한다
accuracies=cross_val_score(s,digit.data,digit.target,cv=5) # 5-겹 교차 검증
# 데이터를 5등분한 뒤, 4덩이로 학습하고 남은 1덩이로 평가하는 과정을 5번 반복한다.
# 모든 샘플이 정확히 한 번씩 테스트에 쓰이므로, 한 번만 분할하는 방식보다 결과가 덜 흔들린다.
# 반환값은 5번의 정확률이 담긴 넘파이 배열이다.

print(accuracies)
# 평균은 성능의 대표값, 표준편차는 분할에 따라 성능이 얼마나 출렁이는지를 나타낸다.
# 표준편차가 크면 그 성능 수치를 그대로 믿기 어렵다는 뜻이다.
print("정확률(평균)=%0.3f, 표준편차 =%0.3f"%(accuracies.mean()*100,accuracies.std()))
# 참고: 평균은 100을 곱해 %로, 표준편차는 곱하지 않아 0~1 단위로 출력된다(단위가 서로 다름).
