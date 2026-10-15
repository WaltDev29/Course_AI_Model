# 영상을 선택해 인식하고 결과를 3개 국어 음성으로 출력하는 GUI 예제
# tkinter는 파이썬에 기본 포함된 GUI 라이브러리이고
# gTTS는 글자를 음성으로 변환하는 구글 서비스다. 실행하려면 인터넷 연결이 필요하다
# winsound와 winsound.Beep는 Windows 전용이므로 다른 운영체제에서는 동작하지 않는다

# Keras가 PyTorch 백엔드로 GPU를 쓰도록 지정한다
# 이 두 줄은 keras를 import 하기 전에 와야 하며 순서가 바뀌면 무시된다
import os
import tempfile
os.environ["KERAS_BACKEND"]="torch"

# 지금 CPU로 도는지 GPU로 도는지 출력한다
# 위의 백엔드 지정이 적용되지 않으면 여기서 드러난다
import keras,torch
print("백엔드:",keras.backend.backend(),"/ 장치:","GPU "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import winsound
from gtts import gTTS
import playsound

# 실행 위치와 상관없이 이 스크립트와 같은 폴더를 기준으로 파일을 찾는다
base_dir=os.path.dirname(os.path.abspath(__file__))
# gTTS가 만드는 임시 음성 파일은 윈도우 임시 폴더에 둔다
# playsound 는 내부에서 윈도우 MCI 를 쓰는데 MCI 가 한글이 들어간 경로를 열지 못한다
# 이 강의자료 경로에는 한글이 있으므로 같은 폴더에 저장하면 재생이 실패한다
mp3_path=os.path.join(tempfile.gettempdir(),'word.mp3')

cnn=keras.models.load_model(os.path.join(base_dir,"my_cnn_for_deploy.h5")) # 학습된 모델 불러오기

class_names_en=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] # CIFAR-10의 부류 이름(영어)
class_names_fr=['avion','voiture','oiseau','chatte','biche','chienne','grenouille','jument','navire','un camion'] # CIFAR-10의 부류 이름(프랑스어)
class_names_de=['Flugzeug','Automobil','Vogel','Katze','Hirsch','Hund','Frosch','Pferd','Schiff','LKW'] # CIFAR-10의 부류 이름(독일어)
class_names_ko = ["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"]

# 여러 함수가 함께 쓰는 값이라 함수 밖에 둔다
# class_id는 마지막 인식 결과의 부류 번호이고, tk_img는 캔버스에 표시 중인 영상이다
# tk_img를 전역으로 두지 않으면 함수가 끝날 때 회수되어 화면에서 영상이 사라진다
class_id=0
tk_img=''

def process_image(): # 사용자가 선택한 영상을 인식하고 결과를 저장
    global class_id, tk_img

    # 화면 표시용 128x128과 인식용 32x32를 각각 만든다
    # 크기가 다른 두 용도를 한 변수로 처리하면 화면이 깨지거나 인식이 실패한다
    fname=filedialog.askopenfilename() # 사용자가 폴더에서 영상을 선택하게 함
    img=Image.open(fname)
    tk_img=img.resize([128,128])
    tk_img=ImageTk.PhotoImage(tk_img)
    canvas.create_image((canvas.winfo_width()/2,canvas.winfo_height()/2),image=tk_img,anchor='center')

    x_test=[]
    x=np.asarray(img.resize([32,32]))/255.0
    x_test.append(x)
    x_test=np.asarray(x_test)
    # 부류 번호 하나로 세 나라 언어의 레이블을 동시에 갱신한다
    # 인식이 끝났음을 소리로 알린다
    res=cnn.predict(x_test) # 영상 인식
    class_id=np.argmax(res)
    label_en['text']='영어: '+class_names_en[class_id] # 레이블 위젯용 텍스트 만들기
    label_fr['text']='프랑스어: '+class_names_fr[class_id]
    label_de['text']='독일어: '+class_names_de[class_id]
    label_ko['text']='한국어: '+class_names_ko[class_id]
    winsound.Beep(frequency=500,duration=250)

# gTTS가 만든 mp3를 파일로 저장한 뒤 재생한다
# 같은 이름의 파일이 남아 있으면 저장이 실패하므로 먼저 삭제한다
def tts_english(): # 영어로 들려주기
    tts=gTTS(text=class_names_en[class_id],lang='en')
    if os.path.isfile(mp3_path): os.remove(mp3_path)
    tts.save(mp3_path)
    playsound.playsound(mp3_path,True)

def tts_french(): # 프랑스어로 들려주기
    tts=gTTS(text=class_names_fr[class_id],lang='fr')
    if os.path.isfile(mp3_path): os.remove(mp3_path)
    tts.save(mp3_path)
    playsound.playsound(mp3_path,True)

def tts_deutsch(): # 독일어로 들려주기
    tts=gTTS(text=class_names_de[class_id],lang='de')
    if os.path.isfile(mp3_path): os.remove(mp3_path)
    tts.save(mp3_path)
    playsound.playsound(mp3_path,True)

def tts_korean(): # 독일어로 들려주기
    tts=gTTS(text=class_names_ko[class_id],lang='ko')
    if os.path.isfile(mp3_path): os.remove(mp3_path)
    tts.save(mp3_path)
    playsound.playsound(mp3_path,True)

def quit_program(): # 프로그램 종료
    win.destroy()

# 창을 만들고 제목과 크기를 지정한다
win=tk.Tk()
win.title('다국어 단어 공부')
win.geometry('512x500')

# command에 함수 이름을 괄호 없이 전달한다
# 괄호를 붙이면 버튼을 만드는 시점에 함수가 즉시 실행되어 버린다
process_button=tk.Button(win,text='영상 선택',command=process_image) # 위젯 생성
quit_button=tk.Button(win,text='끝내기',command=quit_program)
canvas=tk.Canvas(win,width=256,height=256,bg='cyan',bd=4)
label_en=tk.Label(win,width=16,height=1,bg='yellow',bd=4,text='영어',anchor='w')
label_fr=tk.Label(win,width=16,height=1,bg='yellow',bd=4,text='프랑스어',anchor='w')
label_de=tk.Label(win,width=16,height=1,bg='yellow',bd=4,text='독일어',anchor='w')
label_ko=tk.Label(win,width=16,height=1,bg='yellow',bd=4,text='한국어',anchor='w')
tts_en=tk.Button(win,text='듣기',command=tts_english)
tts_fr=tk.Button(win,text='듣기',command=tts_french)
tts_de=tk.Button(win,text='듣기',command=tts_deutsch)
tts_ko=tk.Button(win,text='듣기',command=tts_korean)

# grid는 위젯을 행과 열의 격자에 배치한다. sticky는 칸 안에서의 정렬 방향이다
process_button.grid(row=0,column=0) # 위젯 배치
quit_button.grid(row=1,column=0)
canvas.grid(row=0,column=1)
label_en.grid(row=1,column=1,sticky='e')
label_fr.grid(row=2,column=1,sticky='e')
label_de.grid(row=3,column=1,sticky='e')
label_ko.grid(row=4,column=1,sticky='e')
tts_en.grid(row=1,column=2,sticky='w')
tts_fr.grid(row=2,column=2,sticky='w')
tts_de.grid(row=3,column=2,sticky='w')
tts_ko.grid(row=4,column=2,sticky='w')

# 창을 띄우고 사용자의 입력을 기다린다. 창을 닫을 때까지 여기서 멈춘다
win.mainloop()