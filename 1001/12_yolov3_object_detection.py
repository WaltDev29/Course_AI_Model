# YOLOv3 모델로 영상 속 물체의 위치와 종류를 한꺼번에 검출하는 예제
import numpy as np
import cv2
import os

# OpenCV는 Windows에서 한글이 들어간 경로를 열지 못하므로
# 절대 경로 대신 작업 폴더를 스크립트 위치로 옮긴다
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# YOLO가 구별할 수 있는 80가지 물체 이름을 읽어 옴
# 검출 결과는 이름이 아니라 번호로 나오므로 번호를 이름으로 바꾸는 표가 필요하다
classes = []
f=open('coco.names.txt', 'r')
classes=[line.strip() for line in f.readlines()]
# 부류마다 다른 색을 배정해 상자를 구별하기 쉽게 한다
colors=np.random.uniform(0,255,size=(len(classes),3))

# 영상을 읽고 신경망 입력 형식(블롭)으로 변환
# 화솟값을 0~1로 줄이고 448x448로 크기를 맞추며, OpenCV의 BGR 순서를
# YOLO가 기대하는 RGB 순서로 바꾸라고 swapRB=True를 준다
img=cv2.imread('yolo_test.jpg')
height,width,channels=img.shape
blob=cv2.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True,crop=False)

# 미리 학습된 YOLOv3 모델을 읽어 옴(cfg는 구조, weights는 가중치)
# YOLO는 크기가 다른 물체를 잡으려고 세 곳에서 결과를 내보내므로
# 연결되지 않은 출력층 세 개의 이름을 찾아 둔다
yolo_model=cv2.dnn.readNet('./yolov3.weights','./yolov3.cfg')
layer_names=yolo_model.getLayerNames()
out_layers=[layer_names[i-1] for i in yolo_model.getUnconnectedOutLayers()]

# 신경망을 한 번 통과시켜 세 출력층의 결과를 한꺼번에 받는다
yolo_model.setInput(blob)
output3=yolo_model.forward(out_layers)

# 검출 결과를 해석
# 예측 한 줄은 85개 값이다. 앞 4개는 상자 위치, 5번째는 물체 존재 확률,
# 나머지 80개는 부류별 확률이라 vec85[5:]에서 가장 큰 값을 찾으면 된다
class_ids,confidences,boxes=[],[],[]
for output in output3:
    for vec85 in output:
        scores=vec85[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]
        if confidence>0.5: # 신뢰도가 50% 이상인 경우만 취함
            centerx,centery=int(vec85[0]*width),int(vec85[1]*height)  # [0,1] 표현을 영상 크기로 변환
            w,h=int(vec85[2]*width),int(vec85[3]*height)
            # YOLO는 중심점을 주는데 그리기에는 왼쪽 위 모서리가 필요하다
            x,y=int(centerx-w/2),int(centery-h/2)
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
        
# 같은 물체에 상자가 여러 개 겹쳐 잡히므로 가장 확실한 것만 남긴다(비최대 억제)
indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

# 살아남은 상자만 부류 이름과 신뢰도를 붙여 영상에 그린다
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h=boxes[i]
        text=str(classes[class_ids[i]])+'%.3f'%confidences[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),colors[class_ids[i]],2)
        cv2.putText(img,text,(x,y+30),cv2.FONT_HERSHEY_PLAIN,2,colors[class_ids[i]],2)

# 결과 창을 띄우고 아무 키나 누를 때까지 기다린다
cv2.imshow("Object detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
