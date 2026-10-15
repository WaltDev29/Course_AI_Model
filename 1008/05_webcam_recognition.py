# 웹캠 영상을 입력으로 받아 인식하고 결과를 3개 국어 음성으로 출력하는 GUI 예제
# 7-4와 구조가 같고 입력만 파일에서 웹캠으로 바뀐다
# 웹캠을 다루기 위해 cv2가 추가되었다

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
import cv2

# 7-1이 저장한 모델을 읽어 온다
# 실행 위치와 상관없이 이 스크립트와 같은 폴더를 기준으로 파일을 찾는다
base_dir=os.path.dirname(os.path.abspath(__file__))
# gTTS가 만드는 임시 음성 파일은 윈도우 임시 폴더에 둔다
# playsound 는 내부에서 윈도우 MCI 를 쓰는데 MCI 가 한글이 들어간 경로를 열지 못한다
# 이 강의자료 경로에는 한글이 있으므로 같은 폴더에 저장하면 재생이 실패한다
mp3_path=os.path.join(tempfile.gettempdir(),'word.mp3')

cnn=keras.models.load_model(os.path.join(base_dir,"my_cnn_for_deploy.h5"))

# CIFAR-10의 부류 이름을 영어, 프랑스어, 독일어로 준비한다
class_names_en=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
class_names_fr=['avion','voiture','oiseau','chatte','biche','chienne','grenouille','jument','navire','un camion']
class_names_de=['Flugzeug','Automobil','Vogel','Katze','Hirsch','Hund','Frosch','Pferd','Schiff','LKW']
class_names_ko = ["비행기", "자동차", "새", "고양이", "사슴", "개", "개구리", "말", "배", "트럭"]

# 여러 함수가 함께 쓰는 값이라 함수 밖에 둔다
class_id=0
tk_img=''

# 웹캠을 열어 화면에 계속 보여주다가 ESC를 누른 순간의 화면을 인식한다
def process_video():
    global class_id, tk_img

    # 0은 기본 카메라를 뜻한다. 카메라가 여러 대이면 1, 2로 지정한다
    # waitKey의 27은 ESC 키의 코드값이다
    video=cv2.VideoCapture(0)
    while video.isOpened():
        success,frame=video.read()
        if success:
            cv2.imshow('Camera',frame)
            key=cv2.waitKey(1) & 0xFF
            if key==27:
                break

    video.release()
    cv2.destroyAllWindows()

    # OpenCV는 색을 BGR 순서로 다루고 PIL과 신경망은 RGB 순서를 기대한다
    # 이 변환을 생략하면 빨강과 파랑이 뒤바뀌어 인식률이 크게 낮아진다
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img=Image.fromarray(frame)

    # 화면 표시용 128x128과 인식용 32x32를 각각 만든다
    # 크기가 다른 두 용도를 한 변수로 처리하면 화면이 깨지거나 인식이 실패한다
    tk_img=img.resize([128,128])
    tk_img=ImageTk.PhotoImage(tk_img)
    canvas.create_image((canvas.winfo_width()/2,canvas.winfo_height()/2),image=tk_img,anchor='center')

    # 신경망은 여러 장을 묶은 배치를 기대하므로 한 장이어도 배열로 감싸서 넣는다
    x_test=[]
    x=np.asarray(img.resize([32,32]))/255.0
    x_test.append(x)
    x_test=np.asarray(x_test)
    # 부류 번호 하나로 세 나라 언어의 레이붔을 한번에 갱습한다
    res=cnn.predict(x_test)
    class_id=np.argmax(res)
    label_en['text']='영어: '+class_names_en[class_id]
    label_fr['text']='프랑스어: '+class_names_fr[class_id]
    label_de['text']='독일어: '+class_names_de[class_id]
    label_ko['text']='한국어: '+class_names_ko[class_id]
    winsound.Beep(frequency=500,duration=250)

# gTTS가 만든 mp3를 파일로 저장한 뒤 재생한다
def tts_english():
    tts=gTTS(text=class_names_en[class_id],lang='en')
    if os.path.isfile(mp3_path): os.remove(mp3_path)
    tts.save(mp3_path)
    playsound.playsound(mp3_path,True)

# 프랑스어와 독일어도 언어 코드와 부류 이름표만 다를 바이고 절차는 같다
def tts_french():
    tts=gTTS(text=class_names_fr[class_id],lang='fr')
    if os.path.isfile(mp3_path): os.remove(mp3_path)
    tts.save(mp3_path)
    playsound.playsound(mp3_path,True)

def tts_deutsch():
    tts=gTTS(text=class_names_de[class_id],lang='de')
    if os.path.isfile(mp3_path): os.remove(mp3_path)
    tts.save(mp3_path)
    playsound.playsound(mp3_path,True)

def tts_korean(): # 독일어로 들려주기
    tts=gTTS(text=class_names_ko[class_id],lang='ko')
    if os.path.isfile(mp3_path): os.remove(mp3_path)
    tts.save(mp3_path)
    playsound.playsound(mp3_path,True)

# 창을 닫아 프로길램을 종료한다
def quit_program():
    win.destroy()

# 창과 위젯을 만들어 배치한다. 구성은 7-4와 같고 버튼 이름만 다르다
win=tk.Tk()
win.title('다국어 단어 공부')
win.geometry('512x500')

# command에 함수 이름을 괄호 없이 전달한다
# 괄호를 붙이면 버튼을 만드는 시점에 함수가 직시 실행되어 버린다
process_button=tk.Button(win,text='비디오 선택',command=process_video)
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

# grid는 위젯을 행과 열의 거자에 배정한다. sticky는 칸 안에서의 정렬 방향이다
process_button.grid(row=0,column=0)
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

# 창을 띄우고 사용자의 입력을 기다린다
win.mainloop()