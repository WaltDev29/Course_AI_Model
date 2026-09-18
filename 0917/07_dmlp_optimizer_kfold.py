# 옵티마이저 4종을 5-겹 교차 검증으로 비교하고 결과를 박스플롯으로 나타내는 예제

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD,Adam,Adagrad,RMSprop
from sklearn.model_selection import KFold     # 교차 검증의 데이터 분할은 sklearn의 도구를 빌려 쓴다.

# fashion MNIST를 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000,784)                # 28x28 영상을 784차원 벡터로 펼침
x_test = x_test.reshape(10000,784)
x_train=x_train.astype(np.float32)/255.0            # 0~255를 0~1로 정규화
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)   # 정답을 원핫 코드로 변환
y_test=tf.keras.utils.to_categorical(y_test,10)

# 신경망 구조 설정
n_input=784
n_hidden1=1024
n_hidden2=512
n_hidden3=512
n_hidden4=512
n_output=10

# 하이퍼 매개변수 설정
batch_siz=256
n_epoch=20      # 5-11의 50세대보다 줄였다. 교차 검증은 학습을 여러 번 되풀이하므로 시간을 줄이기 위함이다.
k=5 # 5-겹

# 모델을 설계해주는 함수(모델을 나타내는 객체 model을 반환)
# 교차 검증에서는 겹마다 완전히 새 모델로 시작해야 하므로 이 함수가 반드시 필요하다.
def build_model():
    model=Sequential()
    model.add(Dense(units=n_hidden1,activation='relu',input_shape=(n_input,)))
    model.add(Dense(units=n_hidden2,activation='relu'))
    model.add(Dense(units=n_hidden3,activation='relu'))
    model.add(Dense(units=n_hidden4,activation='relu'))
    model.add(Dense(units=n_output,activation='softmax'))
    return model

# 교차 검증을 해주는 함수(서로 다른 옵티마이저(opt)에 대해)
# 훈련 집합을 5등분해서, 4덩이로 학습하고 남은 1덩이로 평가하는 일을 5번 되풀이한다.
# 교재 원본은 만들어 놓은 옵티마이저 '객체'를 받았으나, Keras 3에서는 옵티마이저 하나를 여러 모델에 재사용할 수 없다.
# 그래서 옵티마이저 '클래스'를 받아 겹마다 opt()로 새 객체를 만들어 쓰도록 바꿨다.
def cross_validation(opt):
    accuracy=[]
    for train_index,val_index in KFold(k).split(x_train):   # 겹마다 훈련용/검증용 인덱스를 돌려준다.
        xtrain,xval=x_train[train_index],x_train[val_index]
        ytrain,yval=y_train[train_index],y_train[val_index]
        dmlp=build_model()      # 이전 겹의 학습 결과가 남지 않도록 매번 새로 만든다.
        dmlp.compile(loss='categorical_crossentropy',optimizer=opt(),metrics=['accuracy'])   # opt() 로 새 옵티마이저 생성
        dmlp.fit(xtrain,ytrain,batch_size=batch_siz,epochs=n_epoch,verbose=1)
        accuracy.append(dmlp.evaluate(xval,yval,verbose=1)[1])  # 그 겹의 검증 정확률을 저장
    return accuracy             # 정확률 5개가 담긴 리스트를 돌려준다.

# 옵티마이저 4개에 대해 교차 검증을 실행
# 옵티마이저 4개 x 5겹 = 모두 20번을 학습하므로 시간이 오래 걸린다.
# 교재 원본은 SGD() 처럼 괄호를 붙여 객체를 넘겼으나, 위 설명대로 이제는 괄호 없이 클래스 자체를 넘긴다.
acc_sgd=cross_validation(SGD)
acc_adam=cross_validation(Adam)
acc_adagrad=cross_validation(Adagrad)
acc_rmsprop=cross_validation(RMSprop)

# 옵티마이저 4개의 정확률을 비교
# 한 번의 실험값이 아니라 5번의 평균이므로 5-11의 결과보다 믿을 만하다.
print("SGD:",np.array(acc_sgd).mean())
print("Adam:",np.array(acc_adam).mean())
print("Adagrad:",np.array(acc_adagrad).mean())
print("RMSprop:",np.array(acc_rmsprop).mean())

import matplotlib.pyplot as plt

# 네 옵티마이저의 정확률을 박스플롯으로 비교
# 상자의 위치는 성능의 높낮이를, 상자의 길이는 5번의 실험이 얼마나 들쭉날쭉했는지를 보여준다.
# 평균만 볼 때와 달리 성능이 안정적인지까지 함께 판단할 수 있다는 것이 박스플롯의 장점이다.
# 교재 원본은 labels= 였으나, matplotlib 3.9부터 이름이 tick_labels로 바뀌어 최신 버전에서는 오류가 난다.
plt.boxplot([acc_sgd,acc_adam,acc_adagrad,acc_rmsprop],tick_labels=["SGD","Adam","Adagrad","RMSprop"])
plt.grid()
plt.show()      
