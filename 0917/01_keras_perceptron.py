# 5-5와 똑같은 퍼셉트론을 Keras 고수준 API로 다시 만든 예제 (코드가 크게 짧아진다)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# OR 데이터 구축
# 교재 원본은 파이썬 리스트를 그대로 fit에 넘겼으나, Keras 3은 리스트를 받지 않아 numpy 배열로 변환한다.
x=np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]],dtype=np.float32)  # 샘플 4개를 모은 4x2 행렬
y=np.array([[-1],[1],[1],[1]],dtype=np.float32)                         # 정답 (OR이므로 (0,0)만 -1)

n_input=2       # 입력 노드 개수 (특징이 2개)
n_output=1      # 출력 노드 개수 (값 하나로 -1인지 1인지를 나타냄)

perceptron=Sequential()     # 층을 순서대로 쌓아 올리는 방식의 모델
# Dense는 완전연결층이다. 5-5에서 손으로 짠 matmul + add + tanh 세 줄이 이 한 줄에 해당한다.
# units는 출력 노드 수, kernel_initializer는 가중치, bias_initializer는 편향의 초기화 방법이다.
# 참고: 최신 Keras에서는 input_shape 대신 Input(shape=(n_input,))을 첫 층으로 두는 방식을 권장하므로 경고가 뜰 수 있다.
perceptron.add(Dense(units=n_output,activation='tanh',input_shape=(n_input,),kernel_initializer='random_uniform',bias_initializer='zeros'))

# compile은 손실 함수와 옵티마이저를 지정해 학습 준비를 마치는 단계이다.
# metrics는 학습 중 지켜볼 지표이며, 손실과 달리 가중치 갱신에는 쓰이지 않는다.
perceptron.compile(loss='mse',optimizer=SGD(learning_rate=0.1),metrics=['mse'])
perceptron.fit(x,y,epochs=500,verbose=2)    # 5-5의 for 반복문 전체가 fit 한 줄로 대체된다.

res=perceptron.predict(x)
print(res)                                  # 5-5와 마찬가지로 -1이나 1에 가까운 실수가 나온다.
