import cv2
import os


'''
This script goes through each video in folder with videos
and saves selected frames to folder with images

'''

videos_path='../dataset/videos'
pathOut='../dataset/images'
frames=10
scale=1


videos=os.listdir(os.path.join(videos_path))

for video in videos:
    print("Currently framing "+video)
    cap = cv2.VideoCapture(os.path.join(videos_path,video))
    count = 0
    count_2 = 0
    
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Save frame
        if ret == True:
            if count%frames==0:
                # Resize frame
                width = int(frame.shape[1] * scale)
                height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
                
                # Save frame
                print('Read %d frame: %s_frame' %(count, ret))
                cv2.imwrite(os.path.join(pathOut, "frame_{}_{:d}.jpg".format(video,count)), frame)  # save frame as JPEG file
                count_2 += 1
            count += 1
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print('Total frames: %d' %count_2)