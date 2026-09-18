# 텐서(tf.Tensor)와 ndarray(numpy)를 섞어서 연산할 수 있음을 보이는 예제

import tensorflow as tf
import numpy as np

t=tf.random.uniform([2,3],0,1)      # TensorFlow로 만든 2x3 텐서 (기본 dtype은 float32)
n=np.random.uniform(0,1,[2,3])      # numpy로 만든 2x3 ndarray (기본 dtype은 float64)
print("tensorflow로 생성한 텐서:\n",t,"\n")
print("numpy로 생성한 ndarray:\n",n,"\n")

# 두 자료형이 달라도 더할 수 있다. ndarray가 자동으로 텐서로 변환되어 계산되기 때문이다.
# 따라서 결과 res는 numpy 배열이 아니라 tf.Tensor 이다.
res=t+n # 텐서 t와 ndarray n의 덧셈
print("덧셈 결과:\n",res)
