import cv2
import numpy as np
import face_recognition

imgCauly = face_recognition.load_image_file("ImagesBasic/cauly.png")
imgCauly = cv2.cvtColor(imgCauly, cv2.COLOR_BGR2RGB)
imgCauly2 = face_recognition.load_image_file("ImagesBasic/Rogerio-ceni.webp")
imgCauly2 = cv2.cvtColor(imgCauly2, cv2.COLOR_BGR2RGB)

