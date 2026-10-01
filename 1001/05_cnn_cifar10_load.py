# 저장된 CNN 모델을 불러와 CIFAR-10 테스트 집합으로 정확률만 평가하는 예제

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
os.environ["KERAS_BACKEND"]="torch"

# 지금 CPU로 도는지 GPU로 도는지 출력한다
# 위의 백엔드 지정이 적용되지 않으면 여기서 드러난다
import keras,torch
print("백엔드:",keras.backend.backend(),"/ 장치:","GPU "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import sys
import numpy as np
from keras.datasets import cifar10

# 신경망 구조와 가중치를 저장하고 있는 파일을 읽어 옴
# 실행 위치와 상관없이 이 스크립트와 같은 폴더의 모델 파일을 가리킴
model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"my_cnn.h5")
# 파일이 없으면 load_model이 알기 어려운 오류를 내므로 미리 확인하고 안내한다
if not os.path.isfile(model_path):
    print("모델 파일을 찾을 수 없습니다:",model_path)
    print("04_cnn_cifar10_save.py를 먼저 실행해 my_cnn.h5를 생성한 뒤 다시 실행하세요.")
    sys.exit(1)

# 구조와 가중치를 한꺼번에 복원하므로 모델을 다시 설계할 필요가 없다
cnn=keras.models.load_model(model_path)
cnn.summary()  # 복원된 층 구성과 매개변수 개수를 확인

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
# 학습할 때와 똑같은 방식으로 전처리해야 정확률이 제대로 나온다
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

# 학습 과정 없이 평가만 수행하므로 몇 초 만에 끝난다
res=cnn.evaluate(x_test,y_test,verbose=0)
print("정확률은",res[1]*100)
