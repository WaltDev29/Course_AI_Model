# 미리 학습된 GloVe 단어 임베딩으로 단어 사이의 관계를 확인하는 예제
import os
import sys

import warnings

# scikit-learn 이 쓰는 joblib 은 물리 코어 수를 알아내려고 wmic 명령을 실행한다
# 최신 Windows 에는 wmic 가 없어 긴 경고가 출력되지만 논리 코어 수로 대체되어 계산은 정상이다
# 수업 중 혼란을 줄이려고 이 경고만 감춘다
warnings.filterwarnings("ignore",message="Could not find the number of physical cores.*")
import numpy as np
from scipy.spatial import distance

# 미리 학습된 GloVe 단어 벡터 파일 읽기
# GloVe는 대규모 말뭉치로 미리 학습된 단어 벡터 모음이다
# 100d는 단어 하나를 100차원 실수 벡터로 표현한다는 뜻이다
# 한 줄이 '단어 수치1 수치2 ... 수치100' 형태다
# 실행 위치와 상관없이 이 스크립트를 기준으로 데이터를 찾는다
# 이 스크립트는 강의자료/SRC/source_1/ch8 에 있고
# 데이터는 강의자료/data/glove.6B 에 있으므로 세 단계 위로 올라간다
base_dir=os.path.dirname(os.path.abspath(__file__))
fname=os.path.normpath(os.path.join(base_dir,'..','..','..','data','glove.6B','glove.6B.100d.txt'))

# 파일이 없으면 원인을 알기 어려운 오류가 발생하므로 먼저 확인한다
if not os.path.isfile(fname):
    print('GloVe 파일을 찾을 수 없습니다:',fname)
    sys.exit(1)
print('GloVe 경로:',fname)
f=open(fname,encoding='utf8')

for line in f: # 내용을 살필 목적으로 첫 번째 단어만 출력
    print(type(line))
    print(line)
    break

# 사전 구축(딕셔너리 자료구조로 표현)
dictionary={}
# 앞에서 한 줄을 이미 읽었으므로 여기서는 두 번째 줄부터 이어서 읽는다
# 줄마다 첫 칸은 단어, 나머지 100칸은 그 단어의 벡터다
for line in f:
    li=line.split()
    word=li[0]
    vector=np.asarray(li[1:],dtype='float32')
    dictionary[word]=vector

# 가장 가까운 단어를 찾아주는 함수
# 주어진 벡터와 유클리드 거리가 가까운 순서로 단어를 늘어놓는다
# 사전 전체와 거리를 계산하므로 한 번 호출에 시간이 걸린다
def find_closest_words(vector):
    return sorted(dictionary.keys(), key=lambda w: distance.euclidean(dictionary[w],vector))

# 가까운 단어 찾기
# 자기 자신이 거리 0이라 항상 첫 번째로 나온다
# 그 뒤에 오는 단어들이 뜻이 가까운 단어다
print(find_closest_words(dictionary['movie'])[:5])
print(find_closest_words(dictionary['school'])[:5])
print(find_closest_words(dictionary['oak'])[:5])

# 단어 추론
# seoul에서 korea를 빼면 '수도'라는 뜻만 남고 거기에 spain을 더하면 madrid가 나온다
# 임베딩 공간에서 뜻의 차이가 벡터의 차이로 나타남을 보여주는 유명한 예다
print(find_closest_words(dictionary["seoul"]-dictionary["korea"]+dictionary["spain"])[:5])
print(find_closest_words(dictionary["animal"]-dictionary["lion"]+dictionary["oak"])[:5])
print(find_closest_words(dictionary["queen"]-dictionary["king"]+dictionary["actress"])[:5])

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# tsne를 이용하여 2차원 공간으로 축소하고 시각화
# 100차원은 그림으로 그릴 수 없으므로 2차원으로 줄인다
# t-SNE는 가까운 것끼리 가깝게 배치하는 데 중점을 두는 축소 방법이다
# 계산량이 커서 앞 100개 단어만 사용한다
tsne=TSNE(n_components=2,random_state=0)
# words와 vectors는 같은 순서이므로 앞 100개를 잘라도 짝이 어긋나지 않는다
words=list(dictionary.keys())
vectors=[dictionary[word] for word in words]
# vectors는 넘파이 배열을 담은 파이썬 리스트다
# 예전 scikit-learn은 리스트도 받았지만 지금은 배열을 요구하므로 변환해서 넘긴다
p2=tsne.fit_transform(np.array(vectors[:100]))
plt.scatter(p2[:,0],p2[:,1])

# 점만 찍으면 어떤 단어인지 알 수 없으므로 이름표를 붙인다
# zip은 짧은 쪽에 맞춰 멈추므로 p2의 100개까지만 표시된다
for label,x,y in zip(words,p2[:,0],p2[:,1]):
    plt.annotate(label,xy=(x,y))

# 주피터가 아니라 .py 로 실행할 때는 이 줄이 있어야 그림 창이 뜬다
plt.show()
