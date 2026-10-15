# YOLO로 웹캠 영상에서 사람을 검출하면 경고음을 내는 예제
import numpy as np
import cv2
import winsound
import os

# OpenCV는 Windows에서 한글이 포함된 경로를 열지 못한다
# 그래서 절대 경로 대신 작업 폴더를 스크립트 위치로 이동한다
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# YOLO가 구별하는 80가지 물체의 이름을 읽어 온다
# 검출 결과는 이름이 아니라 번호로 나오므로 번호를 이름으로 변환할 표가 필요하다
classes = []
f=open('coco.names.txt','r')
classes=[line.strip() for line in f.readlines()]
colors=np.random.uniform(0,255,size=(len(classes),3))

# YOLO는 크기가 다른 물체를 검출하려고 세 곳에서 결과를 출력한다
# 연결되지 않은 출력층 세 개의 이름을 미리 확보한다
yolo_model=cv2.dnn.readNet('yolov3.weights','yolov3.cfg') # 욜로 읽어오기
layer_names=yolo_model.getLayerNames()
# OpenCV 4.5.4부터 getUnconnectedOutLayers가 1차원 배열을 반환하므로 i[0]이 아니라 i를 쓴다
out_layers=[layer_names[i-1] for i in yolo_model.getUnconnectedOutLayers()]

# 6-12는 정지 영상 한 장을 처리하였고, 여기서는 같은 절차를 웹캠 프레임마다 반복한다
# 프레임마다 YOLO를 통과시키므로 사양에 따라 화면이 느려질 수 있다
def process_video(): # 비디오에서 침입자 검출해 알리기
    video=cv2.VideoCapture(0)
    while video.isOpened():
        success,img=video.read()
        if success:
            height,width,channels=img.shape
            blob=cv2.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True,crop=False)

            yolo_model.setInput(blob)
            output3=yolo_model.forward(out_layers)

            # 예측 한 줄은 85개 값이다. 앞 4개는 상자 위치, 5번째는 물체 존재 확률,
            # 나머지 80개는 부류별 확률이므로 vec85[5:]에서 최댓값을 찾는다
            class_ids,confidences,boxes=[],[],[]
            for output in output3:
                for vec85 in output:
                    scores=vec85[5:]
                    class_id=np.argmax(scores)
                    confidence=scores[class_id]
                    if confidence>0.5: # 신뢰도가 50% 이상인 경우만 취함
                        centerx,centery=int(vec85[0]*width),int(vec85[1]*height) # [0,1] 표현을 영상 크기로 변환
                        w,h=int(vec85[2]*width),int(vec85[3]*height)
                        x,y=int(centerx-w/2),int(centery-h/2)
                        boxes.append([x,y,w,h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                    
            # 같은 물체에 상자가 여러 개 겹쳐 검출되므로 가장 확실한 것만 남긴다(비최대 억제)
            indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
                    
            # 남은 상자를 부류 이름과 신뢰도를 붙여 화면에 그린다
            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h=boxes[i]
                    text=str(classes[class_ids[i]])+'%.3f'%confidences[i]
                    cv2.rectangle(img,(x,y),(x+w,y+h),colors[class_ids[i]],2)
                    cv2.putText(img,text,(x,y+30),cv2.FONT_HERSHEY_PLAIN,2,colors[class_ids[i]],2)

            cv2.imshow('Object detection',img)

            # coco.names.txt의 0번이 person이다. 감시 용도이므로 사람만 경고 대상으로 한다
            if 0 in class_ids: # 사람이 검출됨(0='person')
                print('사람이 나타났다!!!')
                winsound.Beep(frequency=2000,duration=500)

        key=cv2.waitKey(1) & 0xFF
        if key==27: break

    video.release()
    cv2.destroyAllWindows()

# ESC를 누를 때까지 계속 감시한다
process_video()