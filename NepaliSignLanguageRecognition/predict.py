import numpy as np
from keras.models import model_from_json
from keras.models import load_model
import operator
import cv2
import sys, os

# Loading the model
classifier = load_model("model2.h5")
# json_file = open("modelw.json", "r")
# model_json = json_file.read()
# json_file.close()
# classifier = model_from_json(model_json)
# # load weights into new model
# classifier.load_weights("modelw.h5")
KA_img = cv2.imread('letter/KA.png')
# ka_img = cv2.cvtColor(ka_img, cv2.COLOR_BGR2GRAY)
# _, ka_mask = cv2.threshold(ka_img, 5, 255, cv2.THRESH_BINARY_INV)
KHA_img = cv2.imread('letter/KHA.png')
GA_img = cv2.imread('letter/GA.png')
GHA_img = cv2.imread('letter/GHA.png')
NGA_img = cv2.imread('letter/NGA.png')
CHA_img = cv2.imread('letter/CHA.png')
CHHA_img = cv2.imread('letter/CHHA.png')
JA_img = cv2.imread('letter/JA.png')
JHA_img = cv2.imread('letter/JHA.png')
YAN_img = cv2.imread('letter/YAN.png')
TA_img = cv2.imread('letter/TA.png')
THA_img = cv2.imread('letter/THA.png')
DA_img = cv2.imread('letter/DA.png')
DHA_img = cv2.imread('letter/DHA.png')
NA_img = cv2.imread('letter/NA.png')
ta_img = cv2.imread('letter/ta.png')
tha_img = cv2.imread('letter/tha.png')
da_img = cv2.imread('letter/da.png')
dha_img = cv2.imread('letter/dha.png')
na_img = cv2.imread('letter/na.png')
PA_img = cv2.imread('letter/PA.png')
PHA_img = cv2.imread('letter/PHA.png')
BA_img = cv2.imread('letter/BA.png')
BHA_img = cv2.imread('letter/BHA.png')
MA_img = cv2.imread('letter/MA.png')
YA_img = cv2.imread('letter/YA.png')
RA_img = cv2.imread('letter/RA.png')
LA_img = cv2.imread('letter/LA.png')
WA_img = cv2.imread('letter/WA.png')
sha_img = cv2.imread('letter/sha.png')
SHA_img = cv2.imread('letter/SHA.png')
SA_img = cv2.imread('letter/SA.png')
HA_img = cv2.imread('letter/HA.png')
KSHA_img = cv2.imread('letter/KSHA.png')
TRA_img = cv2.imread('letter/TRA.png')
GYA_img = cv2.imread('letter/GYA.png')




cap = cv2.VideoCapture(0)

# Category dictionary
# categories = {0: 'क', 1: 'क्ष', 2: 'ख', 3: 'ग', 4: 'घ', 5: 'ङ', 6: 'च',7: 'छ', 8:'ज', 9:'ज्ञ',
#           10: 'झ', 11: 'ञ', 12: 'ट', 13: 'ठ', 14: 'ड', 15:' ढ', 16: 'ण', 17: 'त', 18: 'त्र', 19: 'थ',
#           20: 'द', 21: 'ध', 22: 'न', 23: 'प', 24: 'फ', 25: 'ब', 26: 'भ', 27: 'म', 28: 'य', 29: 'र',
#           30: 'ल', 31: 'व', 32: 'श', 33: 'ष', 34: 'स', 35: 'ह'
#           }
categories = {0: 'ब', 1: 'भ', 2: 'च', 3: 'छ', 4: 'ड', 5: 'द', 6: 'ढ',7: 'ध', 8:'ग', 9:'घ',
        10: 'ज्ञ', 11: 'ह', 12: 'ज', 13: 'झ', 14: 'क', 15:'ख', 16: 'क्ष', 17: 'ल', 18: 'म', 19: 'ण',
        20: 'न', 21: 'ङ', 22: 'प', 23: 'फ', 24: 'र', 25: 'स', 26: 'श', 27: 'ष', 28: 'ट', 29: 'त',
        30: 'ठ', 31: 'थ', 32: 'त्र', 33: 'व', 34: 'य', 35: 'ञ'
        }
#क  ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह क्ष त्र ज्ञ
while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.6*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.4*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]

    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (200, 200 ))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    # roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,999,27)
    cv2.imshow("test", roi)
    # Batch of 1
    result = classifier.predict(roi.reshape(1,200,200,1))
    # result=[[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
    # for key,value in categories.items():
    if result[0][0]==1:
        # final = cv2.bitwise_or(KA_img, KA_img, mask=KA_mask)
        # frame = cv2.add(final, frame)
        cv2.imshow("letter",BA_img)
    elif result[0][1] == 1:
       cv2.imshow("letter",BHA_img)
    elif result[0][2] == 1:
       cv2.imshow("letter",CHA_img)
    elif result[0][3] == 1:
       cv2.imshow("letter",CHHA_img)
    elif result[0][4] == 1:
        cv2.imshow("letter",DA_img)
    elif result[0][5] == 1:
        cv2.imshow("letter",DHA_img)
    elif result[0][6] == 1:
        cv2.imshow("letter",GA_img)
    elif result[0][7] == 1:
        cv2.imshow("letter",GHA_img)
    elif result[0][8] == 1:
        cv2.imshow("letter",GYA_img)
    elif result[0][9] == 1:
        cv2.imshow("letter",HA_img)
    elif result[0][10] == 1:
        cv2.imshow("letter",JA_img)
    elif result[0][11] == 1:
        cv2.imshow("letter",JHA_img)
    elif result[0][12] == 1:
        cv2.imshow("letter",KA_img)
    elif result[0][13] == 1:
        cv2.imshow("letter",KHA_img)
    elif result[0][14] == 1:
        cv2.imshow("letter",KSHA_img)
    elif result[0][15] == 1:
        cv2.imshow("letter",LA_img)
    elif result[0][16] == 1:
        cv2.imshow("letter",MA_img)
    elif result[0][17] == 1:
        cv2.imshow("letter",NA_img)
    elif result[0][18] == 1:
        cv2.imshow("letter",NGA_img)
    elif result[0][19] == 1:
        cv2.imshow("letter",PA_img)
    elif result[0][20] == 1:
        cv2.imshow("letter",PHA_img)
    elif result[0][21] == 1:
        cv2.imshow("letter",RA_img)
    elif result[0][22] == 1:
        cv2.imshow("letter",SA_img)
    elif result[0][23] == 1:
        cv2.imshow("letter",SHA_img)
    elif result[0][24] == 1:
        cv2.imshow("letter",TA_img)
    elif result[0][25] == 1:
        cv2.imshow("letter",THA_img)
    elif result[0][26] == 1:
        cv2.imshow("letter",TRA_img)
    elif result[0][27] == 1:
        cv2.imshow("letter",WA_img)
    elif result[0][28] == 1:
        cv2.imshow("letter",YA_img)
    elif result[0][29] == 1:
        cv2.imshow("letter",YAN_img)
    elif result[0][30] == 1:
        cv2.imshow("letter",da_img)
    elif result[0][31] == 1:
        cv2.imshow("letter",dha_img)
    elif result[0][32] == 1:
        cv2.imshow("letter",na_img)
    elif result[0][33] == 1:
        cv2.imshow("letter",sha_img)
    elif result[0][34] == 1:
        cv2.imshow("letter",ta_img)
    elif result[0][35] == 1:
        cv2.imshow("letter",tha_img)
    # Displaying the predictions
    cv2.imshow("Frame", frame)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break


cap.release()
cv2.destroyAllWindows()
