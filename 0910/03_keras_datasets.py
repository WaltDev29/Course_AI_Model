# Keras에 내장된 대표 데이터셋 4종을 읽어 텐서 모양(shape)을 확인하는 예제

import tensorflow as tf
import tensorflow.keras.datasets as ds

# MNIST 읽고 텐서 모양 출력
# load_data()는 (훈련 입력, 훈련 정답), (테스트 입력, 테스트 정답)을 한 번에 돌려준다.
# 처음 실행할 때만 인터넷에서 내려받고, 이후에는 사용자 폴더의 .keras/datasets 에서 읽는다.
(x_train, y_train),(x_test, y_test)=ds.mnist.load_data()    # 28x28 손글씨 숫자 (훈련 6만 장, 테스트 1만 장)
yy_train=tf.one_hot(y_train,10,dtype=tf.int8) # 원핫 코드로 변환
# 원핫 코드는 정답 3을 [0,0,0,1,0,0,0,0,0,0]처럼 부류 개수만큼의 0/1 벡터로 바꾼 것이다.
# 신경망의 출력층이 부류마다 노드 하나씩을 가지므로 정답도 같은 모양으로 맞춰 주는 것이다.
print("MNIST: ",x_train.shape,y_train.shape,yy_train.shape)     # (60000,28,28) (60000,) (60000,10)

# CIFAR-10 읽고 텐서 모양 출력
(x_train,y_train),(x_test,y_test)=ds.cifar10.load_data()        # 32x32 컬러 영상 10부류 (훈련 5만 장)
yy_train=tf.one_hot(y_train,10,dtype=tf.int8)
# 주의: CIFAR-10의 정답은 (50000,1) 모양이라 원핫 결과가 (50000,10)이 아니라 (50000,1,10)이 된다.
# MNIST와 정답의 모양이 다르기 때문이며, 실제로 학습에 쓸 때는 차원을 정리해 주어야 한다.
print("CIFAR-10: ",x_train.shape,y_train.shape,yy_train.shape)

# Boston Housing 읽고 텐서 모양 출력
(x_train,y_train),(x_test,y_test)=ds.boston_housing.load_data() # 특징 13개로 집값을 맞히는 회귀 데이터
# 부류를 고르는 문제가 아니라 실수값을 예측하는 문제이므로 원핫 코드가 필요 없다.
print("Boston Housing: ",x_train.shape,y_train.shape)

# Reuters 읽고 텐서 모양 출력
(x_train,y_train),(x_test,y_test)=ds.reuters.load_data()        # 뉴스 기사를 46개 주제로 분류하는 텍스트 데이터
# 기사마다 길이가 다르므로 x_train은 (8982,28,28) 같은 직육면체가 아니라 (8982,) 모양의 리스트 배열이다.
# 텍스트를 신경망에 넣으려면 길이를 맞추는 별도의 전처리가 필요하다는 뜻이다.
print("Reuters: ",x_train.shape,y_train.shape)
