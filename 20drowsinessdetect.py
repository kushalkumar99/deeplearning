from scipy.spatial import distance as dt
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound
freq = 2500
duration = 1000

def eyeAspectRatio(eye):
    A = dt.euclidean(eye[1],eye[5])
    B = dt.euclidean(eye[2],eye[4])
    C = dt.euclidean(eye[0],eye[3])
    ear = (A+B)/(2.0*C)
    return ear
blink = 0
count = 0
earthresh = 0.3#distance between two lides of eyes
#if earthresh<0.3 drowsiness detected
earframes = 48#for how many frames drowsiness will be detected
shapepredictor = "shape_predictor_68_face_landmarks.dat"

cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapepredictor)

#getting the coordinates of left and right
(lstart,lend) = face_utils.FACIAL_LANDMARKS_5_IDXS["left_eye"]
(rstart,rend) = face_utils.FACIAL_LANDMARKS_5_IDXS["right_eye"]

while True:
    _,frame = cam.read()
    frame = imutils.resize(frame,width=450)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray,0)
    
    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)#converting into an array by appending coords 
        lefteye = shape[lstart:lend]
        righteye = shape[rstart:rend]
        leftEAR = eyeAspectRatio(lefteye)
        rightEAR = eyeAspectRatio(righteye)
        
        ear = (lefteye+rightEAR)/2.0
        lefteyehull = cv2.convexHull(lefteye)
        righteyehull = cv2.convexHull(righteye)
        cv2.drawContours(frame,[lefteyehull],-1(0,0,255),1)
        cv2.drawContours(frame,[righteyehull],-1(0,0,255),1)
        
        if ear<earthresh:
            count+=1
            
            if count>=earframes:
                cv2.putText(frame,"DROWSINESS DETECTED",(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(0,255,0),2)
                winsound.Beep(freq,duration)
        
        elif count>=1 and count<=earframes:
            blink+=1
            cv2.putText(frame,"BLINK {}".format(int(blink)),(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(0,0,255,),2)
        else:
            count=0
        
    
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1)
    if key ==27:
        break
cam.release()
cv2.destroyAllWindows()