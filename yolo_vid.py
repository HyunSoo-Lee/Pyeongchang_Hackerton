import cv2
import numpy as np
import time # -- 프레임 계산을 위해 사용
import matplotlib.pyplot as plt
import math

vedio_path = './situation.mp4' #-- 사용할 영상 경로
min_confidence = 0.7

red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)

def detectAndDisplay(frame):
    #녹화된 동영상 전처리
    start_time = time.time()
    img = cv2.resize(frame, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape

    #-- 창 크기 설정
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    #-- 탐지한 객체의 클래스 예측 
    class_ids = []
    confidences = []
    boxes = []

    #거리 측정용 list
    distance = [[],[]]
    j = 0

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                # 탐지한 객체 박싱
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
               
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]], confidences[i]*100)
            print(i, label)
            color = colors[i] #-- 경계 상자 컬러 설정 / 단일 생상 사용시 (255,255,255)사용(B,G,R)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
            if ("person" in label ) or ("backpack" in label):
                cv2.line(img, ((x+(x+w))//2, (y+(y+h))//2), ((x+(x+w))//2, (y+(y+h))//2), (0,0,255), 5)
                distance[j].append((x+(x+w))//2)
                distance[j].append((y+(y+h))//2)
                j += 1
    j = 0
    if distance[1]:
        length_a = (distance[0][0] - distance[1][0]) * (distance[0][0] - distance[1][0])
        length_b = (distance[0][1] - distance[1][1]) * (distance[0][1] - distance[1][1])
        length_sum = math.sqrt(length_a + length_b)
        length_text = "Distance is " + str(length_sum)
        if length_sum >= 550:
            cv2.line(img, distance[0], distance[1], green, 3)
            print("Status : Safe Distance")
        elif length_sum >= 400 and length_sum < 550:
            cv2.line(img, distance[0], distance[1], blue, 3)
            print("Status : Average Distance")
        else:
            cv2.line(img, distance[0], distance[1], red, 3)
            print("Status : Caution! \nStay away from the machine!")
        cv2.putText(img, length_text, (0,30), font, 1, (0,0,255), 1)
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    cv2.imshow("YOLO test", img)
    
#-- yolo 포맷 및 클래스명 불러오기
model_file = './yolov3.weights' #-- 본인 개발 환경에 맞게 변경할 것
config_file = './yolov3.cfg' #-- 본인 개발 환경에 맞게 변경할 것
net = cv2.dnn.readNet(model_file, config_file)

#-- 클래스(names파일) 오픈 / 본인 개발 환경에 맞게 변경할 것
classes = []
with open("./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#-- 비디오 활성화
cap = cv2.VideoCapture(vedio_path) #-- 웹캠 사용시 vedio_path를 0 으로 변경
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    #-- q 입력시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()