import cv2
import numpy as np
import face_recognition

imgCauly = face_recognition.load_image_file("ImagesBasic/cauly.png")
imgCauly = cv2.cvtColor(imgCauly, cv2.COLOR_BGR2RGB)
imgCauly2 = face_recognition.load_image_file("ImagesBasic/Rogerio-ceni.webp")
imgCauly2 = cv2.cvtColor(imgCauly2, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgCauly)[0]
encodeCauly = face_recognition.face_encodings(imgCauly)[0]
cv2.rectangle(imgCauly,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgCauly2)[0]
encodeCauly2 = face_recognition.face_encodings(imgCauly2)[0]
cv2.rectangle(imgCauly2,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

resultado = face_recognition.compare_faces([encodeCauly],encodeCauly2)
faceDis = face_recognition.face_distance([encodeCauly],encodeCauly2)
print(resultado, faceDis)
cv2.putText(imgCauly2,f'{resultado} {round(faceDis[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Cauly", imgCauly)
cv2.imshow("Cauly 2", imgCauly2)
cv2.waitKey(0)

