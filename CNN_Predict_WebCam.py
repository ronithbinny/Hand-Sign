import cv2
import numpy as np
from keras.models import load_model

classifier = load_model("cnn_adv_01.h5")

# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# From Web Cam :
cap = cv2.VideoCapture(0)

while True :
    
    _,frame = cap.read()
    
    frame=cv2.flip(frame,1)
    
    roi = frame[100:300, 50:250]
    
    cv2.rectangle(frame,(50,100),(250,300),(0,255,0),0)
    
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
 
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    edged = cv2.Canny(gray, 30, 200)
     
    edged = cv2.resize(edged,(64,64))
    
    fin = edged.reshape(1,64,64,1)
      
    X = classifier.predict((np.asarray(fin)))
    
    pro = classifier.predict_proba((np.asarray(fin)))
    
    classes = classifier.predict_classes((np.asarray(fin)))
    
    signs = ["Cool", "Hello", "Hold", "Peace", "Point"]
    
    frame = cv2.putText(frame, signs[classes[0]], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    
    cv2.imshow("Frames",frame)
    
    cv2.imshow("Edged", edged)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()