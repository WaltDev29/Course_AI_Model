# TensorFlow 설치 확인과 텐서 생성 기초 (버전 출력과 난수 텐서 만들기)

import tensorflow as tf

print(tf.__version__)               # 설치된 TensorFlow 버전 확인 (설치가 제대로 됐는지 보는 용도)
a=tf.random.uniform([2,3],0,1)      # 0~1 사이 균등 난수로 2행 3열 텐서 생성
print(a)                            # 값뿐 아니라 shape와 dtype이 함께 출력된다.
print(type(a))                      # EagerTensor로 출력된다. tf.Tensor의 한 종류이며 numpy의 ndarray에 대응하는 개념이다.
