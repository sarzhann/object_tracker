from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import cv2
import sys
import random
import numpy as np
from sort import *

#model and labels details
model_path = 'models/primesource.pth'
label_path = 'models/labels_prime_source.txt'


cap = cv2.VideoCapture(sys.argv[1]) if len(sys.argv) ==2 else cv2.VideoCapture(0) # capture from file


#Write video to file
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

#labels
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

#Geting list of colors
random.seed(25)
colors=[(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for x in range(num_classes)]

#fixed colors
black=(0,0,0)
white=(255,255,255)
red=(0,0,255)
green=(0,255,0)
blue=(255,0,0)


#Loading model
net = create_mobilenetv1_ssd(num_classes, is_test=True)
net.load(model_path)
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

#function for checking is object below or above line
def check_position(x,y,x1,y1,x2,y2):
    # x,y - coordinates of point on frame
    # x1,y1,x2,y2 - coordinates of line
    return (y1-y2)*x+(x2-x1)*y+(x1*y2-x2*y1)
    
#checking object inside rectangle:
def check_inside(x,y, x1,y1, x2,y2, x3,y3, x4,y4):
    # x,y - coordinates of point on frame
    # x1,y1, x2,y2, x3,y3, x4,y4 - coordinates of nodes of rectangle
    position1=check_position(x,y, x1,y1,x2,y2)
    position2=check_position(x,y, x2,y2,x3,y3)
    position3=check_position(x,y, x4,y4,x3,y3)
    position4=check_position(x,y, x1,y1,x4,y4)
    if (position1>0)&(position2>0)&(position3<0)&(position4<0):
        return True
    else: return False

#drawing polygons
def draw_poly(points):
    cv2.polylines(orig_image, [np.array([points])], 1, red, 1)

x1_wp,y1_wp, x2_wp,y2_wp, x3_wp,y3_wp, x4_wp,y4_wp = 155,200, 448,200, 390,386, 100,384 #position of manager's work place
x1_cl,y1_cl, x2_cl,y2_cl = 10,400, 1200,400 #position of crossing line


count_up=0
count_down=0
frame=0
object_ids=[]
mot_tracker=Sort()

data={}
data2={}

while(cap.isOpened()):

    ret, orig_image = cap.read()
    if ret == True:
        cv2.rectangle(orig_image, (7,5,220,55),white, -1) #white box
        cv2.rectangle(orig_image, (7,5,220,55),black, 1) #black rectangle
        
        draw_poly([[x1_wp, y1_wp], [x2_wp, y2_wp], [x3_wp, y3_wp], [x4_wp, y4_wp]]) #polygon workplace
        #draw_poly([[x1_cl, y1_cl], [x2_cl, y2_cl]]) #polygon crossline

        boxes, labels, probs = predictor.predict(orig_image, 10, 0.4) #detection
        tracked_objects = mot_tracker.update(boxes.numpy()) #tracking

        for i, detections in enumerate(tracked_objects):
            object_class=class_names[labels[i]]
            color=colors[labels[i]]
            box = boxes[i]
            obj_id = detections[4]

            # Draw bbox of object
            label = f"{object_class}: {probs[i]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 1)
            cv2.rectangle(orig_image, (box[0], box[1]), (box[0]+110, box[1]-30), color, -1)
            cv2.putText(orig_image, label, (box[0], box[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1) 

            # Put Object id
            cv2.putText(orig_image, "object id {}".format(obj_id), (box[0], box[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1)

            # Draw center
            x_center = (box[0]+box[2])//2
            y_center = (box[1]+box[3])//2
            cv2.circle(orig_image, (x_center,y_center), 1, color, 2)
            

            # Check position of person in workplace
            is_inside_workplace = check_inside(x_center, y_center, x1_wp,y1_wp, x2_wp,y2_wp, x3_wp,y3_wp, x4_wp,y4_wp)
            if is_inside_workplace == True:
                cv2.putText(orig_image, 'Manager is at workplace', (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1)
               

            
            #Check crossing line
            object_position=check_position(x_center, y_center, x1_cl,y1_cl, x2_cl,y2_cl)
            if (data.get(obj_id,0)>0)&(object_position.item()<0):
                data2[obj_id]='in'
            elif (data.get(obj_id,0)<0)&(object_position.item()>0):
                data2[obj_id]='out'
            data[obj_id]=object_position
            result=data2.get(obj_id)
            

            # Count objects
            if (obj_id not in object_ids)&(result == 'in'):
                object_ids.append(obj_id)
                count_up+=1
            elif (obj_id not in object_ids)&(result == 'out'):
                object_ids.append(obj_id)
                count_down+=1
                    



        cv2.putText(orig_image, 'in: {}'.format(count_up), (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1)
        cv2.putText(orig_image, 'out: {}'.format(count_down), (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1)        
        cv2.imshow('object tracking', orig_image)

        out.write(orig_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()