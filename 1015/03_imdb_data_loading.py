# IMDB 영화평 원본 데이터를 폴더에서 직접 읽어 내용을 확인하는 예제
import os
import sys

# 원본 IMDB 데이터 읽기
# IMDB 원본은 긍정(pos)과 부정(neg) 폴더로 나뉘어 있고 평 하나가 txt 파일 하나다
# 폴더 이름이 그대로 정답 레이블이 된다
# 실행 위치와 상관없이 이 스크립트를 기준으로 데이터셋을 찾는다
# 이 스크립트는 강의자료/SRC/source_1/ch8 에 있고
# 데이터셋은 강의자료/data/aclImdb 에 있으므로 세 단계 위로 올라간다
base_dir=os.path.dirname(os.path.abspath(__file__))
directory=os.path.normpath(os.path.join(base_dir,'..','..','..','data','aclImdb','aclImdb','test'))

# 폴더가 없으면 원인을 알기 어려운 오류가 발생하므로 먼저 확인한다
if not os.path.isdir(directory):
    print('IMDB 데이터 폴더를 찾을 수 없습니다:',directory)
    sys.exit(1)
print('데이터 경로:',directory)
x=[];y=[]
# neg를 먼저 읽으므로 x의 앞쪽은 모두 부정, 뒤쪽은 모두 긍정이다
# 이대로 학습하면 안 되고 섞어야 한다는 점을 14행 출력으로 확인할 수 있다
# encoding='utf8'을 지정하지 않으면 Windows에서 읽기 오류가 발생한다
for c in ['neg','pos']:
    curr=os.path.join(directory,c)
    for fname in os.listdir(curr):
        if fname[-4:]=='.txt':
            f=open(os.path.join(curr,fname),encoding='utf8')
            x.append(f.read())
            y.append(c)
            
print("첫 번째 댓글:",x[0]); print("첫 번째 평가:",y[0])
print("마지막 댓글:",x[-1]); print("마지막 평가:",y[-1])