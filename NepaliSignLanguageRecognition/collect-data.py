import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/क")
    os.makedirs("data/train/ख")
    os.makedirs("data/train/ग")
    os.makedirs("data/train/घ")
    os.makedirs("data/train/ङ")
    os.makedirs("data/train/च")
    os.makedirs("data/train/छ")
    os.makedirs("data/train/ज")
    os.makedirs("data/train/झ")
    os.makedirs("data/train/ञ")
    os.makedirs("data/train/ट")
    os.makedirs("data/train/ठ")
    os.makedirs("data/train/ड")
    os.makedirs("data/train/ढ")
    os.makedirs("data/train/ण")
    os.makedirs("data/train/त")
    os.makedirs("data/train/थ")
    os.makedirs("data/train/द")
    os.makedirs("data/train/ध")
    os.makedirs("data/train/न")
    os.makedirs("data/train/प")
    os.makedirs("data/train/फ")
    os.makedirs("data/train/ब")
    os.makedirs("data/train/भ")
    os.makedirs("data/train/म")
    os.makedirs("data/train/य")
    os.makedirs("data/train/र")
    os.makedirs("data/train/ल")
    os.makedirs("data/train/व")
    os.makedirs("data/train/श")
    os.makedirs("data/train/ष")
    os.makedirs("data/train/स")
    os.makedirs("data/train/ह")
    os.makedirs("data/train/क्ष")
    os.makedirs("data/train/त्र")
    os.makedirs("data/train/ज्ञ")

    os.makedirs("data/test/क")
    os.makedirs("data/test/ख")
    os.makedirs("data/test/ग")
    os.makedirs("data/test/घ")
    os.makedirs("data/test/ङ")
    os.makedirs("data/test/च")
    os.makedirs("data/test/छ")
    os.makedirs("data/test/ज")
    os.makedirs("data/test/झ")
    os.makedirs("data/test/ञ")
    os.makedirs("data/test/ट")
    os.makedirs("data/test/ठ")
    os.makedirs("data/test/ड")
    os.makedirs("data/test/ढ")
    os.makedirs("data/test/ण")
    os.makedirs("data/test/त")
    os.makedirs("data/test/थ")
    os.makedirs("data/test/द")
    os.makedirs("data/test/ध")
    os.makedirs("data/test/न")
    os.makedirs("data/test/प")
    os.makedirs("data/test/फ")
    os.makedirs("data/test/ब")
    os.makedirs("data/test/भ")
    os.makedirs("data/test/म")
    os.makedirs("data/test/य")
    os.makedirs("data/test/र")
    os.makedirs("data/test/ल")
    os.makedirs("data/test/व")
    os.makedirs("data/test/श")
    os.makedirs("data/test/ष")
    os.makedirs("data/test/स")
    os.makedirs("data/test/ह")
    os.makedirs("data/test/क्ष")
    os.makedirs("data/test/त्र")
    os.makedirs("data/test/ज्ञ")


# Train or test
mode = 'train'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Getting count of existing images
    count = {'क': len(os.listdir(directory+"/क")),
    'ख': len(os.listdir(directory+"/ख")),
    'ग': len(os.listdir(directory+"/ग")),
    'घ': len(os.listdir(directory+"/घ")),
    'ङ': len(os.listdir(directory+"/ङ")),
    'च': len(os.listdir(directory+"/च")),
    'छ': len(os.listdir(directory+"/छ")),
    'ज': len(os.listdir(directory+"/ज")),
    'झ': len(os.listdir(directory+"/झ")),
    'ञ': len(os.listdir(directory+"/ञ")),
    'ट': len(os.listdir(directory+"/ट")),
    'ठ': len(os.listdir(directory+"/ठ")),
    'ड': len(os.listdir(directory+"/ड")),
    'ढ': len(os.listdir(directory+"/ढ")),
    'ण': len(os.listdir(directory+"/ण")),
    'त': len(os.listdir(directory+"/त")),
    'थ': len(os.listdir(directory+"/थ")),
    'द': len(os.listdir(directory+"/द")),
    'ध': len(os.listdir(directory+"/ध")),
    'न': len(os.listdir(directory+"/न")),
    'प': len(os.listdir(directory+"/प")),
    'फ': len(os.listdir(directory+"/फ")),
    'ब': len(os.listdir(directory+"/ब")),
    'भ': len(os.listdir(directory+"/भ")),
    'म': len(os.listdir(directory+"/म")),
    'य': len(os.listdir(directory+"/य")),
    'र': len(os.listdir(directory+"/र")),
    'ल': len(os.listdir(directory+"/ल")),
    'व': len(os.listdir(directory+"/व")),
    'श': len(os.listdir(directory+"/श")),
    'ष': len(os.listdir(directory+"/ष")),
    'स': len(os.listdir(directory+"/स")),
    'ह': len(os.listdir(directory+"/ह")),
    'क्ष': len(os.listdir(directory+"/क्ष")),
    'त्र': len(os.listdir(directory+"/त्र")),
    'ज्ञ': len(os.listdir(directory+"/ज्ञ"))}



    # fontpath="./preeti.TTF"
    # font=ImageFont.truetype(fontpath,32)
    # img_pil=Image.fromarray(frame)
    # draw=ImageDraw.Draw(img_pil)
    # draw.text((50,80),"क",font=font,fill=(0,255,0,0))
    # frame=np.array(img_pil)

    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "क: "+str(count['क']), (10, 440), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ख : "+str(count['ख']), (10, 460), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ग : "+str(count['ग']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "घ : "+str(count['घ']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ङ : "+str(count['ङ']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "च : "+str(count['च']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "छ : "+str(count['छ']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ज : "+str(count['ज']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "झ : "+str(count['झ']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ञ : "+str(count['ञ']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ट : "+str(count['ट']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ठ : "+str(count['ठ']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ड : "+str(count['ड']), (10, 360), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ढ : "+str(count['ढ']), (10, 380), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ण : "+str(count['ण']), (10, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "त : "+str(count['त']), (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "थ : "+str(count['थ']), (10, 440), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "द : "+str(count['द']), (10, 460), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ध : "+str(count['ध']), (100, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "न : "+str(count['न']), (100, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "प : "+str(count['प']), (100, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "फ : "+str(count['फ']), (100, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ब : "+str(count['ब']), (100, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "भ : "+str(count['भ']), (100, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "म : "+str(count['म']), (100, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "य : "+str(count['य']), (100, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "र : "+str(count['र']), (100, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ल : "+str(count['ल']), (100, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "व : "+str(count['व']), (100, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "श : "+str(count['श']), (100, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ष : "+str(count['ष']), (100, 360), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "स : "+str(count['स']), (100, 380), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ह : "+str(count['ह']), (100, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "क्ष : "+str(count['क्ष']), (100, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "त्र : "+str(count['त्र']), (100, 440), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "ज्ञ : "+str(count['ज्ञ']), (100, 460), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)


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
    roi = cv2.resize(roi, (200, 200))

    cv2.imshow("Frame", frame)

    #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(mask, kernel, iterations=1)
    #img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY_INV)
    # roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,999,27)
    # roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,255,27)
    cv2.imshow("ROI", roi)

    interrupt = cv2.waitKey(1)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(directory+'क/'+str(count['क'])+'.jpg', roi)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory+'ख/'+str(count['ख'])+'.jpg', roi)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'ग/'+str(count['ग'])+'.jpg', roi)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory+'घ/'+str(count['घ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'ङ/'+str(count['ङ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory+'च/'+str(count['च'])+'.jpg', roi)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory+'छ/'+str(count['छ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'ज/'+str(count['ज'])+'.jpg', roi)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory+'झ/'+str(count['झ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory+'ञ/'+str(count['ञ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'ट/'+str(count['ट'])+'.jpg', roi)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory+'ठ/'+str(count['ठ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'ड/'+str(count['ड'])+'.jpg', roi)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory+'ढ/'+str(count['ढ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'ण/'+str(count['ण'])+'.jpg', roi)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'त/'+str(count['त'])+'.jpg', roi)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory+'थ/'+str(count['थ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(directory+'द/'+str(count['द'])+'.jpg', roi)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory+'ध/'+str(count['ध'])+'.jpg', roi)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(directory+'न/'+str(count['न'])+'.jpg', roi)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(directory+'प/'+str(count['प'])+'.jpg', roi)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'फ/'+str(count['फ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory+'ब/'+str(count['ब'])+'.jpg', roi)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'भ/'+str(count['भ'])+'.jpg', roi)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'म/'+str(count['म'])+'.jpg', roi)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(directory+'य/'+str(count['य'])+'.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'र/'+str(count['र'])+'.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'ल/'+str(count['ल'])+'.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'व/'+str(count['व'])+'.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'श/'+str(count['श'])+'.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'ष/'+str(count['ष'])+'.jpg', roi)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory+'स/'+str(count['स'])+'.jpg', roi)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory+'ह/'+str(count['ह'])+'.jpg', roi)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory+'क्ष/'+str(count['क्ष'])+'.jpg', roi)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory+'त्र/'+str(count['त्र'])+'.jpg', roi)
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'ज्ञ/'+str(count['ज्ञ'])+'.jpg', roi)

cap.release()
cv2.destroyAllWindows()
"""
d = "old-data/test/0"
newd = "data/test/0"
for walk in os.walk(d):
    for file in walk[2]:
        roi = cv2.imread(d+"/"+file)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(newd+"/"+file, mask)
"""
