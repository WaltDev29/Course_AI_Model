# 인식 결과에 따라 영상을 부류별 폴더로 자동 분류해 저장하는 예제

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

cnn=keras.models.load_model(os.path.join(base_dir,'my_cnn_for_deploy.h5')) # 학습된 모델 불러오기
class_names=['airplane','automobile','bird','cat','deer','dog','flog','horse','ship','truck'] # CIFAR-10의 부류 이름

# 원본 영상(img_orig)과 파일 이름(fname)도 함께 보관한다
# 분류한 뒤 저장할 때는 32x32로 축소한 것이 아니라 원본을 써야 하기 때문이다
x_test=[]
img_orig=[]
fname=[]
for filename in os.listdir(test_dir): # 폴더에서 테스트 영상 읽기
    if 'jpg' not in filename:
        continue
    img=Image.open(os.path.join(test_dir,filename))
    img_orig.append(img)
    fname.append(filename)
    x=np.asarray(img.resize([32,32]))/255.0
    x_test.append(x)
x_test=np.asarray(x_test)

pred=cnn.predict(x_test) # 예측

# test_images/class_buckets 아래에 부류마다 폴더를 생성한다
# os.chdir로 작업 폴더를 옮겨 다니므로 실행 도중 오류가 발생하면
# 작업 폴더가 하위 폴더에 남는다. 이때는 프로그램을 다시 실행하면 된다
os.chdir(test_dir)
if not os.path.isdir('class_buckets'):
    os.mkdir('class_buckets')
os.chdir('class_buckets')
for i in range(len(class_names)): # 부류별로 폴더 만들기
    if not os.path.isdir(class_names[i]):
        os.mkdir(class_names[i])

# argmax는 확률이 가장 높은 부류의 번호를 돌려준다
# 그 번호로 폴더 이름을 정하고 원본 영상을 같은 파일 이름으로 저장한다
for i in range(len(x_test)): # 인식 결과에 따라 폴더에 저장
    folder_name=class_names[np.argmax(pred[i])]
    os.chdir(folder_name)
    img_orig[i].save(fname[i])
    os.chdir('..')