# Python program to implement
# WebCam Motion Detector
import face_recognition
from playsound import playsound
import cv2
import time
import os
from time import gmtime, strftime
from datetime import datetime

# Assigning our static_back to None
static_back = None

# List when any moving object appear
motion_list = [ None, None ]

# Time of movement
timing = []

known_image=[]
# Enter your Known and Unknown Face Directories here
#[!] Needs to be customised according to your computer directories
Known_Faces_Path="/Users/mohitpandrangi/Desktop/OpenCvFaceDetection/Known_Faces"
Unknown_Faces_Path="/Users/mohitpandrangi/Desktop/OpenCvFaceDetection/Unkown_Faces"
known_face_encodings=[]
for image_path in os.listdir("/Users/mohitpandrangi/Desktop/OpenCvFaceDetection/Known_Faces"):
    print('Image Path',image_path)
    if(image_path!='.DS_Store'):
        known_image.append(face_recognition.load_image_file(Known_Faces_Path+'/'+image_path))
for i in known_image:
    known_face_encodings.append(face_recognition.face_encodings(i)[0])

known_face_names = [
    "Murthy",
    "Sandhya",
    "Mohit",
    "Shreya",
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
previousname=''
# Capturing video
video = cv2.VideoCapture(0)
video.set(3,640)
video.set(4,480)
timestr=strftime('/Users/mohitpandrangi/Desktop/OpenCvFaceDetection/Motion Log/Video Logs'+'/'+"%Y_%m_%d", gmtime())+'.avi'
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter(timestr,fourcc, 20.0, (640,480))
# Infinite while loop to treat stack of image as video
start=0.0
while True:
    # Reading frame(image) from video
    check, frame = video.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Initializing motion = 0(no motion)
    motion = 0
    # Converting color image to gray_scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Converting gray scale image to GaussianBlur
    # so that change can be find easily
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings,face_encoding,tolerance=0.56)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                if(name!=previousname):
                    message="Detected:"+name
                    print("Detected:",name)
                    previousname=name

            face_names.append(name)
            if(name=="Unknown"):
                cv2.imwrite(strftime(Unknown_Faces_Path+'/'+"%Y_%m_%d_%H%M", gmtime())+'Unkwown.jpg',frame)
                print("Saved to Unknown")

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    # In first iteration we assign the value of static_back to our first frame
    if static_back is None:
        static_back = gray
        continue
    # Difference between static background and current frame(which is GaussianBlur)
    diff_frame = cv2.absdiff(static_back, gray)
    # If change in between static background and current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, 10, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 3)
    equ=cv2.equalizeHist(diff_frame)

    # Finding contour of moving object
    ( cnts, _) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 100000:
            continue
        motion = 1
        font=cv2.FONT_HERSHEY_SIMPLEX
        (x, y, w, h) = cv2.boundingRect(contour)
        # making green rectangle arround the moving object
        cv2.putText(frame,"Motion Detected",(x+w,y+h),font,0.5,(0,0,255),3,cv2.LINE_AA)
        print('Motion Detected:',x+w,y+h,':Time:',start)
        start+=time.time()
        # plays sound when motion is detected
        playsound('MotionDetected.mp3')
    if(motion==1):
        out.write(frame)
    if(motion==1 and len(face_encodings)>0):
            # Writes file to Motion Log Folder
            #[!] Change the directory according your computer
            cv2.imwrite(strftime('/Users/mohitpandrangi/Desktop/OpenCvFaceDetection/Motion Log/Photos'+'/'+"%Y_%m_%d_%H%M", gmtime())+'.jpg',frame)
    # Appending status of motion
    static_back=gray
    motion_list.append(motion)

    motion_list = motion_list[-2:]

    # Appending Start time of motion
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        timing.append(datetime.now())

        # Appending End time of motion
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        timing.append(datetime.now())
    # Displaying color frame with contour of motion of object
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    # if q entered whole process will stop
    if key == ord('q'):
        # if something is movingthen it append the end time of movement
        if motion == 1:
            timing.append(datetime.now())
        break
video.release()
out.release()
# Destroying all the windows
cv2.destroyAllWindows()