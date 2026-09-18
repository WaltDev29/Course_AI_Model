# 저수준 TensorFlow API로 퍼셉트론을 직접 학습시키는 예제 (경사 하강법의 동작을 눈으로 확인)

import tensorflow as tf

# OR 데이터 구축
x=[[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]     # 샘플 4개를 모은 4x2 행렬
y=[[-1],[1],[1],[1]]                            # 정답 (OR이므로 (0,0)만 -1)

# 가중치 초기화
# 5-4와 달리 정답 가중치를 직접 주지 않고, 난수에서 출발해 학습으로 찾아간다.
w=tf.Variable(tf.random.uniform([2,1],-0.5,0.5))    # -0.5~0.5 사이 난수로 초기화
b=tf.Variable(tf.zeros([1]))                        # 편향은 0에서 시작

# 옵티마이저
opt=tf.keras.optimizers.SGD(learning_rate=0.1)      # 확률적 경사 하강법. 학습률 0.1은 한 번에 얼마나 이동할지를 정한다.

# 전방 계산(식 (4.3))
def forward():
    s=tf.add(tf.matmul(x,w),b)  # 가중치 합
    o=tf.tanh(s)                # 활성 함수. 5-4의 sign과 달리 미분이 가능해서 학습에 쓸 수 있다.
    return o

# 손실 함수 정의
def loss():
    o=forward()
    return tf.reduce_mean((y-o)**2) # 평균제곱오차(MSE). 예측과 정답의 차이를 제곱해 평균낸 값이다.

# 500세대까지 학습(100세대마다 학습 정보 출력)
# 교재 원본은 opt.minimize(loss, var_list=[w,b]) 한 줄이었으나, Keras 3에서 옵티마이저의 minimize가 없어져 아래와 같이 바꿨다.
# GradientTape는 그 블록 안에서 일어난 계산을 기록해 두었다가 나중에 기울기를 꺼내 쓸 수 있게 해주는 도구이다.
for i in range(500):
    with tf.GradientTape() as tape:
        lv=loss()                           # 손실을 구하는 계산 과정을 tape에 기록
    grads=tape.gradient(lv,[w,b])           # 기록을 되짚어 w와 b에 대한 기울기(미분값)를 구함
    opt.apply_gradients(zip(grads,[w,b]))   # 기울기의 반대 방향으로 w와 b를 한 번 갱신
    if(i%100==0): print('loss at epoch',i,'=',lv.numpy())   # .numpy()는 텐서에서 실제 값을 꺼내는 함수이다.

# 학습된 퍼셉트론으로 OR 데이터를 예측
o=forward()
print(o)    # tanh의 출력이므로 정확히 -1과 1이 아니라 그에 가까운 실수가 나온다.
