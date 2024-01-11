import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import json
with open('mdrive.json') as f:
 data = json.load(f)

import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive','https://www.googleapis.com/auth/drive.file','https://www.googleapis.com/auth/spreadsheets']

creds = ServiceAccountCredentials.from_json_keyfile_name('mdrive.json', scope)

client = gspread.authorize(creds)

sheet = client.open("ClassroomStatus").sheet1


 
video_capture = cv2.VideoCapture(0)
 
shivans_image = face_recognition.load_image_file("data/shivans1.jpeg")
shivans_encoding = face_recognition.face_encodings(shivans_image)[0]

umang_image = face_recognition.load_image_file("data/umang.jpeg")
umang_encoding = face_recognition.face_encodings(umang_image)[0]
  
prat_image = face_recognition.load_image_file("data/prat.jpeg")
prat_encoding = face_recognition.face_encodings(prat_image)[0]
    
mali_image = face_recognition.load_image_file("data/mali.jpeg")
mali_encoding = face_recognition.face_encodings(mali_image)[0]
    
  
known_face_encoding = [
shivans_encoding,
umang_encoding,
prat_encoding,
mali_encoding
]
 
known_faces_names = [
"Shivans",
"Umang",
"Pratham",
"Sudipta"
]
 
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
 
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    #rgb_small_frame = small_frame[:,:,::-1]
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (5,50)
                fontScale              = 1
                fontColor              = (255,0,0)
                thickness              = 2
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 



video_capture.release()
cv2.destroyAllWindows()
f.close()

f = open(current_date+'.csv','r')
values = [r for r in csv.reader(f)]
sheet.update(values)
print("Data Sent")
f.close()





