# 저장된 모델로 폴더 안의 영상을 인식하고 부류별 확률을 막대그래프로 표시하는 예제

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
os.environ["KERAS_BACKEND"]="torch"

# 지금 CPU로 도는지 GPU로 도는지 출력한다
# 위의 백엔드 지정이 적용되지 않으면 여기서 드러난다
import keras,torch
print("백엔드:",keras.backend.backend(),"/ 장치:","GPU "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import numpy as np
from PIL import Image

# 실행 위치와 상관없이 이 스크립트와 같은 폴더를 기준으로 파일을 찾는다
base_dir=os.path.dirname(os.path.abspath(__file__))
test_dir=os.path.join(base_dir,'test_images')

cnn=keras.models.load_model(os.path.join(base_dir,"my_cnn_for_deploy.h5")) # 학습된 모델 불러오기
class_names=['airplane','automobile','bird','cat','deer','dog','flog','horse','ship','truck'] # CIFAR-10의 부류 이름

# test_images 폴더의 jpg 파일을 모두 읽어 신경망 입력 형태로 변환한다
# CIFAR-10으로 학습한 모델이므로 어떤 크기의 영상이든 32x32로 축소해야 한다
x_test=[]
for filename in os.listdir(test_dir): # 폴더에서 테스트 영상 읽기
    if 'jpg' not in filename:
        continue
    img=Image.open(os.path.join(test_dir,filename))
    x=np.asarray(img.resize([32,32]))/255.0
    x_test.append(x)
x_test=np.asarray(x_test)

# 예측 결과는 영상마다 10개의 확률값이다. 합이 1이 되도록 softmax가 정규화한 값이다
pred=cnn.predict(x_test) # 예측

import matplotlib.pyplot as plt

# 위쪽 줄에 원본 영상을, 아래쪽 줄에 부류별 확률 막대그래프를 배치한다
# 확률이 한 부류에 몰려 있으면 확신이 높은 인식이고, 여러 부류에 퍼져 있으면 애매한 인식이다
n=len(x_test)
plt.figure(figsize=(18,4))

for i in range(n):
    plt.subplot(2,n,i+1)
    plt.imshow(x_test[i])
    plt.xticks([]);plt.yticks([])
    plt.subplot(2,n,n+i+1)
    # 첫 영상에만 부류 이름을 붙인다. 나머지는 약자로 표시해 그림이 좁아지지 않게 한다
    if i==0:
        plt.barh(class_names,pred[i])
    else:
        plt.barh(['a','A','b','c','d','D','f','h','s','t'],pred[i])
    plt.xticks([])
    
    
plt.show()