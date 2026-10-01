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
from sklearn.model_selection import KFold
import sys
import time
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False    # 마이너스 기호 깨짐 방지


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

# # ImageNet 100만 장으로 학습된 ResNet50을 특징 추출기로 가져온다
# # include_top=False는 원래의 1000부류 분류층을 떼어낸다는 뜻이다
# # 그 자리에 우리 부류 수에 맞는 새 분류층을 붙이는 것이 전이 학습이다
# base_model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
# cnn=Sequential()
# cnn.add(base_model)
# cnn.add(Flatten())
# cnn.add(Dense(1024,activation='relu'))
# cnn.add(Dense(no_class,activation='softmax'))

# # 학습률을 0.00002로 아주 작게 준다
# # 기본값을 쓰면 애써 학습된 ResNet50의 가중치가 첫 몇 배치 만에 망가지기 때문이다
# cnn.compile(loss='categorical_crossentropy',optimizer=Adam(0.00002),metrics=['accuracy'])
# hist=cnn.fit(x_train,y_train,batch_size=16,epochs=10,validation_data=(x_test,y_test),verbose=1)

# # 영상 수백 장만으로도 높은 정확률이 나오는 것이 전이 학습의 장점이다
# # evaluate는 batch_size를 주지 않으면 32를 쓴다
# # 8GB VRAM에서는 224x224 ResNet50을 32장씩 처리할 수 없으므로 학습과 같은 16으로 맞춘다
# res=cnn.evaluate(x_test,y_test,batch_size=16,verbose=0)
# print("정확률은",res[1]*100)


k=3
img_size = 224
batch_siz = 16
n_epoch = 5
n_done = 0

# ====== Time Formatting ======
def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}분 {s}초" if m > 0 else f"{s:.1f}초"

# ====== Cross Validation Function ======
def cross_validation(use_pretrained, freeze, learning_rate):
    global n_done  # 함수 밖의 진행 상황 변수를 갱신한다

    accuracy = []
    loss = []

    t_config = time.time()  # 이 조합을 시작한 시각

    for train_index, val_index in KFold(n_splits=k, shuffle=True, random_state=42).split(x_train):

        t_fold = time.time()  # 이 겹을 시작한 시각

        xtrain, xval = x_train[train_index], x_train[val_index]
        ytrain, yval = y_train[train_index], y_train[val_index]

        # 학습이 길어서 지금 어느 지점인지 먼저 알리고 시작한다
        n_done += 1
        print(
            "  겹", len(accuracy) + 1, "/", k,
            "  (전체", n_done, "/", k * n_config, "번째 학습)"
        )

        # weights='imagenet'이면 ImageNet 100만 장으로 학습된 가중치에서 출발하고(전이 학습)
        # weights=None이면 똑같은 구조를 난수로 초기화해 처음부터 학습한다
        base_model = ResNet50(
            weights='imagenet' if use_pretrained else None,
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )

        # trainable=False로 얼리면 ResNet50 부분은 그대로 두고 새로 붙인 분류기만 학습한다
        # 학습할 가중치가 확 줄어 빠르고, 데이터가 적을 때 과잉적합도 덜하다
        base_model.trainable = not freeze

        # 겹마다 새 모델을 만들어야 한다. 그러지 않으면 앞 겹에서 학습한 가중치가 남아
        # 검증 집합을 이미 본 셈이 되기 때문이다
        cnn = Sequential()

        cnn.add(base_model)
        cnn.add(Flatten())
        cnn.add(Dense(1024, activation='relu'))
        cnn.add(Dense(no_class, activation='softmax'))

        cnn.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate),
            metrics=['accuracy']
        )

        # verbose=1이면 에포크마다 진행 막대와 손실/정확률이 실시간으로 그려진다
        # 한 줄씩만 보려면 verbose=2, 조용히 돌리려면 verbose=0으로 바꾼다
        cnn.fit(
            xtrain,
            ytrain,
            batch_size=batch_siz,
            epochs=n_epoch,
            verbose=1
        )

        # evaluate는 [손실, 정확률] 두 개를 돌려준다. 6-11은 정확률만 썼지만 여기서는 둘 다 받는다
        # 손실은 교차 엔트로피(식 5.10)이므로 정답에 배정한 확률이 1에 가까울수록 0에 가까워진다
        score = cnn.evaluate(xval, yval, verbose=0)

        loss.append(score[0])
        accuracy.append(score[1])

        print(
            "  겹", len(accuracy),
            "정확률", round(accuracy[-1] * 100, 2),
            "손실", round(loss[-1], 4),
            "/", fmt(time.time() - t_fold),
            "걸림"
        )

    # 남은 조합이 얼마나 걸릴지 가늠하도록 조합단위 시간도 찍는다
    print(
        "  -> 이 조합 총", fmt(time.time() - t_config),
        "/ 시작 후 누적", fmt(time.time() - t_start)
    )

    return accuracy, loss  # 겹마다의 정확률 k개와 손실 k개를 모아 돌려준다


t_start = time.time()
n_config = 5


# 실습 1
FF00002 = cross_validation(False, False, 0.00002)  # 처음부터 학습
TT00002 = cross_validation(True, True, 0.00002) # 전이학습

# 실습 2
TT001 = cross_validation(True, True, 0.001) # 전이학습, freeze
TF001 = cross_validation(True, False, 0.001) # 전이학습, unfreeze
TF00002 = cross_validation(True, False, 0.00002) # 전이학습, unfreeze


# ====== 실습 1 시각화: 정확률 & 손실 박스플롯 ======
labels = ["Transfer Learning", "From Scratch"]
plt.figure(figsize=(12, 5))

# 정확률(Accuracy) 박스플롯
plt.subplot(1, 2, 1)
plt.grid(True)
plt.boxplot([TT00002[0], FF00002[0]], tick_labels=labels)
plt.title("실습 1: 정확률 (Accuracy) 비교")
plt.ylabel("Accuracy")

# 손실(Loss) 박스플롯
plt.subplot(1, 2, 2)
plt.grid(True)
plt.boxplot([TT00002[1], FF00002[1]], tick_labels=labels)
plt.title("실습 1: 손실 (Loss) 비교")
plt.ylabel("Loss")

plt.tight_layout()

# ====== 실습 2 시각화: 정확률 & 손실 박스플롯 ======
labels = ["TT0.001", "TT0.00002", "TF0.001", "TF0.00002"]
plt.figure(figsize=(12, 5))

# 정확률(Accuracy) 박스플롯
plt.subplot(1, 2, 1)
plt.grid(True)
plt.boxplot([TT001[0], TT00002[0], TF001[0], TF00002[0]], tick_labels=labels)
plt.title("실습 2: 정확률 (Accuracy) 비교")
plt.ylabel("Accuracy")

# 손실(Loss) 박스플롯
plt.subplot(1, 2, 2)
plt.grid(True)
plt.boxplot([TT001[1], TT00002[1], TF001[1], TF00002[1]], tick_labels=labels)
plt.title("실습 2: 손실 (Loss) 비교")
plt.ylabel("Loss")

plt.tight_layout()

# ====== 실습 3: 정확률-손실 평면에서 최적 조합 선택 ======
models = {
    "Freeze lr=0.001": TT001,
    "Freeze lr=0.00002": TT00002,
    "Fine-tune lr=0.001": TF001,
    "Fine-tune lr=0.00002": TF00002,
}

# 각 조합의 k-겹 평균 정확률과 평균 손실 계산
results = {}
for name, data in models.items():
    mean_acc = np.mean(data[0])
    mean_loss = np.mean(data[1])
    results[name] = (mean_loss, mean_acc)
plt.figure(figsize=(8, 6))
plt.grid(True, linestyle='-', alpha=0.5)

# 4개 모델 점 찍기 및 레이블 표시
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 파랑, 주황, 초록, 빨강
for idx, (name, (loss_val, acc_val)) in enumerate(results.items()):
    plt.scatter(loss_val, acc_val, s=120, color=colors[idx], zorder=3)
    # 점 옆에 텍스트 라벨 표시
    plt.text(loss_val + 0.3, acc_val + 0.01, name, fontsize=9, verticalalignment='bottom')

# 최적 조합 찾기 (정확률 최고 조합)
best_name = max(results, key=lambda k: results[k][1])
best_loss, best_acc = results[best_name]

# 최적 조합 강조 링 (빨간색 동심원)
plt.scatter(best_loss, best_acc, s=400, facecolors='none', edgecolors='red', linewidth=2.5, zorder=4)
plt.title("실습 3: best combination = upper left")
plt.xlabel("Mean loss")
plt.ylabel("Mean accuracy")
plt.tight_layout()
plt.show()
