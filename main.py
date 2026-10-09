import cv2
import os
import numpy as np
folder = "known faces"
faces = []
labels=[]
names={}

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
label_id=0
for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    image= cv2.imread(path)
    if image is None:
        continue
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected = detector.detectMultiScale(gray, 1.3, 5)

    if len(detected)==0:
        continue
    x,y,w,h = detected[0]
    face = cv2.resize(gray[y:y+h, x:x+w], (200,200))
    faces.append(face)
    labels.append(label_id)
    names[label_id] = os.path.splitext(filename)[0].replace("_", " ")
    label_id +=1
if not faces:
    print("No faces found in the dictionary ,buddy!")
    raise SystemExit
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
image_path = input("Enter the path of the image: ")
image = cv2.imread(image_path)
if image is None:
    print("Image not FOund, Check the file Path")
    raise SystemExit
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detected = detector.detectMultiScale(gray, 1.3,5)
if len(detected) == 0:
    print("No face detected in the input image")
else:
    for x,y,w,h in detected:
        face=cv2.resize(gray[y:y+h, x:x+w], (200,200))
        label, confidence = recognizer.predict(face)
        if confidence < 65:
            name = names[label]
            result = f"{name} (match candidate)"
        else:
            result="Unknown Person"
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2) 
        cv2.putText(image, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        print("Recognition result:", result)
    cv2.imshow("Face Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()