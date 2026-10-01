# ImageDataGenerator가 만들어내는 증대 영상을 눈으로 확인하는 예제

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
os.environ["KERAS_BACKEND"]="torch"

# 지금 CPU로 도는지 GPU로 도는지 출력한다
# 위의 백엔드 지정이 적용되지 않으면 여기서 드러난다
import keras,torch
print("백엔드:",keras.backend.backend(),"/ 장치:","GPU "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

from keras.datasets import cifar10
# Keras 3에서 ImageDataGenerator는 legacy 모듈로 이동하였다
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# CIFAR-10의 부류 이름
# 레이블이 0~9 숫자로 저장돼 있어 이름을 찾는 표로 쓴다
class_names=['airplane','automobile','bird','cat','deer','dog','flog','horse','ship','truck']

# CIFAR-10 데이터셋을 읽고 신경망에 입력할 형태로 변환
(x_train, y_train), (x_test, y_test)=cifar10.load_data()
x_train=x_train.astype('float32'); x_train/=255
x_train=x_train[0:12,]; y_train=y_train[0:12,] # 앞 12개에 대해서만 증대 적용

# 앞 12개 영상을 그려줌
# 아래 증대 결과와 비교할 원본이다
plt.figure(figsize=(16,2))
plt.suptitle("First 12 images in the train set")
for i in range(12):
    plt.subplot(1,12,i+1)
    plt.imshow(x_train[i])
    plt.xticks([]); plt.yticks([])
    # y_train[i]는 [3]처럼 원소 1개짜리 배열이므로 [0]으로 숫자를 꺼내야 한다
    plt.title(class_names[int(y_train[i][0])])

# 영상 증대기 생성
# 회전, 좌우/상하 이동, 좌우 반전을 무작위로 섞어 훈련 집합을 부풀린다
# 상하 반전을 쓰지 않는 이유는 뒤집힌 자동차나 새는 현실에 없기 때문이다
batch_siz=6 # 한 번에 생성하는 양
generator=ImageDataGenerator(rotation_range=30.0,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
# flow는 배열에서 배치를 무한히 만들어내는 반복자를 돌려준다
gen=generator.flow(x_train,y_train,batch_size=batch_siz)

# 첫 번째 증대하고 그리기
# next()를 호출할 때마다 원본을 무작위로 변형한 새 배치가 나온다
img,label=next(gen)
plt.figure(figsize=(16,3))
plt.suptitle("Generatior trial 1")
for i in range(batch_siz):
    plt.subplot(1,batch_siz,i+1)
    plt.imshow(img[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[int(label[i][0])])

plt.show()

# 두 번째 증대하고 그리기
# 같은 원본인데도 변형이 매번 달라지는 것을 1번 결과와 비교해 확인한다
img,label=next(gen)
plt.figure(figsize=(16,3))
plt.suptitle("Generatior trial 2")
for i in range(batch_siz):
    plt.subplot(1,batch_siz,i+1)
    plt.imshow(img[i])
    plt.xticks([]); plt.yticks([])
    plt.title(class_names[int(label[i][0])])
plt.show()
