from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import cv2
import time
import random
import numpy as np
from sort import *
from datetime import datetime

#model and labels details
model_path = 'models/mobilenet-v1-ssd-mp-0_675.pth'
label_path = 'models/labels_voc.txt'
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

#Loading model and tracker
net = create_mobilenetv1_ssd(num_classes, is_test=True)
net.load(model_path)
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
mot_tracker=Sort()

#Colors
black=(0,0,0)
white=(255,255,255)
red=(0,0,255)
green=(0,255,0)
blue=(255,0,0)

#OpenCV
write_video=True
video_length=10
cap = cv2.VideoCapture(0)

max_people=0
initial_time=time.time()
start_time=-11
video_counter=0

while(cap.isOpened()):
    
    if write_video&(time.time()-start_time>video_length):
        current_time = datetime.now().strftime("%d.%M.%Y_%H:%M:%S")
        start_time=time.time()
        saved_video=False
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/Videos/video_record_{}.avi'.format(current_time), fourcc, 11.0, (int(cap.get(3)), int(cap.get(4))))

    ret, orig_image = cap.read()
    orig_image = cv2.flip(orig_image, 1)

    if ret == True:
        boxes, labels, probs = predictor.predict(orig_image, 10, 0.4) #detection
        tracked_objects = mot_tracker.update(boxes.numpy()) #tracking
        people_count=0

        for i, detections in enumerate(tracked_objects):
            object_class=class_names[labels[i]]
            if object_class=='person':
                people_count+=1
                box = boxes[i]
                obj_id = detections[4]

                # Draw bbox of object
                cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), green, 1)
                cv2.rectangle(orig_image, (box[0], box[1]), (box[0]+105, box[1]-20), green, -1)
                cv2.putText(orig_image, " {}: {:.1f}%".format(object_class, probs[i]*100), (box[0], box[1]-6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, black, 1) 

        if people_count > max_people:
            max_people=people_count
            cv2.imwrite("output/Images/New_maximum_{}.jpg".format(people_count), orig_image)
            print('image saved')
 

        cv2.rectangle(orig_image, (7,5,205,45),white, -1) #white box
        cv2.rectangle(orig_image, (7,5,205,45),black, 1) #black rectangle
        cv2.putText(orig_image, "Currently on camera: {}".format(people_count), (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black,1)
        cv2.putText(orig_image, "Record: {}".format(max_people), (15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black,1)   

        cv2.imshow('Tracker', orig_image)
        if write_video:
            out.write(orig_image)
            if time.time()-start_time>video_length:
                out.release()
                print('video recordered')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print("total working time: {:.2f} seconds".format((time.time()-initial_time)))