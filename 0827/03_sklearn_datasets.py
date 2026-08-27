# sklearn 내장/외부 데이터셋(숫자 영상, 얼굴 영상, 뉴스 문서)을 읽고 살펴보는 예제

from sklearn import datasets          # sklearn이 제공하는 예제 데이터셋 모음
import matplotlib.pyplot as plt       # 영상과 그래프를 화면에 그리는 라이브러리

# [1] 필기 숫자 데이터셋(digits): 8x8 크기의 흑백 숫자 영상 1,797장
#     load_* 계열 함수는 sklearn에 함께 설치된 작은 데이터를 즉시 불러온다.
digit=datasets.load_digits()

plt.figure(figsize=(5,5))                                                 # 그림 창 크기를 5x5인치로 지정
plt.imshow(digit.images[0],cmap=plt.cm.gray_r,interpolation='nearest')    # gray_r: 값이 클수록 검게(반전 흑백), nearest: 화소를 보간 없이 그대로 확대
plt.show()                                                                # 0번 샘플을 그림
print(digit.data[0]) # 0번 샘플의 화솟값을 출력
# images는 8x8 2차원 배열, data는 이를 64개로 펼친 1차원 배열이다. 같은 영상의 두 가지 표현이다.
print("이 숫자는 ",digit.target[0],"입니다.")                              # target: 정답 레이블(0~9)

# [2] LFW 얼굴 데이터셋: 유명인 얼굴 사진 모음
#     fetch_* 계열 함수는 인터넷에서 내려받아 캐시에 저장하므로 첫 실행 시 시간이 걸린다.
#     min_faces_per_person=70 : 사진이 70장 이상인 인물만 사용
#     resize=0.4               : 원본을 40% 크기로 줄여 메모리와 계산량을 절약
lfw=datasets.fetch_lfw_people(min_faces_per_person=70,resize=0.4) # 데이터셋 읽기

plt.figure(figsize=(20,5))            # 얼굴 8장을 가로로 늘어놓기 위해 넓은 창을 만든다

for i in range(8): # 처음 8명을 디스플레이
    plt.subplot(1,8,i+1)                                  # 1행 8열 격자에서 i+1번째 칸을 선택(칸 번호는 1부터 시작)
    plt.imshow(lfw.images[i],cmap=plt.cm.bone)            # bone: 푸른빛이 도는 흑백 컬러맵
    plt.title(lfw.target_names[lfw.target[i]])            # target은 인물 번호, target_names로 실제 이름을 찾아 제목으로 단다

plt.show()

# [3] 20 newsgroups 데이터셋: 20개 주제로 분류된 뉴스그룹 게시글(텍스트)
#     subset='train' : 전체 중 훈련용으로 나뉜 부분만 읽는다
news=datasets.fetch_20newsgroups(subset='train') # 데이터셋 읽기
print("*****\n",news.data[0],"\n*****") # 0번 샘플 출력
# 영상 데이터와 달리 data가 문자열이므로, 학습에 쓰려면 별도의 특징 추출 과정이 필요하다.
print("이 문서의 부류는 <",news.target_names[news.target[0]],"> 입니다.")   # 숫자 레이블을 주제 이름으로 변환해 출력
