# ImageNet으로 미리 학습된 ResNet50을 전이 학습해 CUB200 새 영상을 분류하는 예제

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
os.environ["KERAS_BACKEND"]="torch"

# 지금 CPU로 도는지 GPU로 도는지 출력한다
# 위의 백엔드 지정이 적용되지 않으면 여기서 드러난다
import keras,torch
print("백엔드:",keras.backend.backend(),"/ 장치:","GPU "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import numpy as np
from keras.models import Sequential
from keras.layers import Flatten,Dense
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.utils import load_img,img_to_array
import sys

# 실행 위치와 상관없이 이 스크립트를 기준으로 데이터셋을 찾는다
# 이 스크립트는 강의자료/SRC/source_1/ch6 에 있고
# 데이터셋은 강의자료/data/CUB200 에 있으므로 세 단계 위로 올라간다
base_dir=os.path.dirname(os.path.abspath(__file__))
data_dir=os.path.normpath(os.path.join(base_dir,'..','data','CUB200'))
train_folder=os.path.join(data_dir,'train')
test_folder=os.path.join(data_dir,'test')

# 폴더가 없으면 원인을 알기 어려운 오류가 발생하므로 먼저 확인한다
if not os.path.isdir(train_folder):
    print("데이터셋 폴더를 찾을 수 없습니다:",train_folder)
    sys.exit(1)
print("데이터셋 경로:",data_dir)

# CUB200은 새 200종을 담은 데이터셋이라 전부 쓰면 메모리와 시간이 많이 든다
class_reduce=0.1 # 부류 수 줄여서 데이터양 줄임(속도와 메모리 효율을 위해)
no_class=int(len(os.listdir(train_folder))*class_reduce) # 부류 개수

# 훈련 영상을 폴더에서 직접 읽어 배열로 만든다
# CIFAR-10처럼 준비된 데이터셋이 아니라 부류마다 하위 폴더로 나뉘어 있어
# 폴더를 순회하며 영상을 읽고 폴더 순번 i를 레이블로 붙여야 한다
x_train,y_train=[],[]
for i,class_name in enumerate(os.listdir(train_folder)):
    if i<no_class: # 13~14행이 지정한 부류만 사용
        for fname in os.listdir(train_folder+'/'+class_name):
            # ResNet50의 입력 규격인 224x224로 맞춰 읽는다
            img=load_img(train_folder+'/'+class_name+'/'+fname,target_size=(224,224))
            # 흑백이나 투명 채널이 섞인 영상은 채널 수가 3이 아니라 건너뛴다
            if len(img.getbands())!=3:
                print("주의: 유효하지 않은 영상 발생",class_name,fname)
                continue
            x=img_to_array(img)
            # ResNet50이 학습될 때 쓴 것과 같은 방식으로 화솟값을 변환해야 한다
            x=preprocess_input(x)
            x_train.append(x)
            y_train.append(i)

# 테스트 영상도 같은 방식으로 읽는다
x_test,y_test=[],[]
for i,class_name in enumerate(os.listdir(test_folder)):
    if i<no_class: # 13~14행이 지정한 부류만 사용
        for fname in os.listdir(test_folder+'/'+class_name):
            img=load_img(test_folder+'/'+class_name+'/'+fname,target_size=(224,224))
            if len(img.getbands())!=3:
                print("주의: 유효하지 않은 영상 발생",class_name,fname)
                continue
            x=img_to_array(img)
            x=preprocess_input(x)
            x_test.append(x)
            y_test.append(i)

# 파이썬 리스트를 신경망이 받을 수 있는 넘파이 배열로 바꾼다
x_train=np.asarray(x_train)
y_train=np.asarray(y_train)
x_test=np.asarray(x_test)
y_test=np.asarray(y_test)
y_train=keras.utils.to_categorical(y_train,no_class)
y_test=keras.utils.to_categorical(y_test,no_class)

# ImageNet 100만 장으로 학습된 ResNet50을 특징 추출기로 가져온다
# include_top=False는 원래의 1000부류 분류층을 떼어낸다는 뜻이다
# 그 자리에 우리 부류 수에 맞는 새 분류층을 붙이는 것이 전이 학습이다
base_model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
cnn=Sequential()
cnn.add(base_model)
cnn.add(Flatten())
cnn.add(Dense(1024,activation='relu'))
cnn.add(Dense(no_class,activation='softmax'))

# 학습률을 0.00002로 아주 작게 준다
# 기본값을 쓰면 애써 학습된 ResNet50의 가중치가 첫 몇 배치 만에 망가지기 때문이다
cnn.compile(loss='categorical_crossentropy',optimizer=Adam(0.00002),metrics=['accuracy'])
hist=cnn.fit(x_train,y_train,batch_size=16,epochs=10,validation_data=(x_test,y_test),verbose=1)

# 영상 수백 장만으로도 높은 정확률이 나오는 것이 전이 학습의 장점이다
# evaluate는 batch_size를 주지 않으면 32를 쓴다
# 8GB VRAM에서는 224x224 ResNet50을 32장씩 처리할 수 없으므로 학습과 같은 16으로 맞춘다
res=cnn.evaluate(x_test,y_test,batch_size=16,verbose=0)
print("정확률은",res[1]*100)