import cv2
import os

cap = cv2.VideoCapture(0)

hand_name = input("Enter Name Of Hand Gestuer : ")

print("Show the Hand Sign in the green box and press ENTER for taking the next photo, press Q to EXIT.")

os.mkdir(hand_name)

x = 1
i = 0

while True :
    
    _,frame = cap.read()
    
    frame=cv2.flip(frame,1)
    
    roi = frame[100:300, 50:250]
    
    cv2.rectangle(frame,(50,100),(250,300),(0,255,0),0)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(gray, 30, 200)
    
    cv2.imshow("Frames",frame)
    
    cv2.imshow("Edged", edged)
    
    if i in range(200,1000,2) :
    
        filename = "{}/{}-{}.jpg".format(hand_name,hand_name,i)
        cv2.imwrite(filename,edged)
        print("Image - ",x)
        x = x + 1
    
    i = i + 1
    
    if i == 150 :
        print("Be ready! Starting of clicking photos ")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if i >= 1001 :
        break


cap.release()
cv2.destroyAllWindows()