import cv2
import face_recognition as fr

#Setting up of the camera and the image details
height=720
width=1280
cam=cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS,30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

#Training of our images
dadFace=fr.load_image_file('C:/Users/User/Documents/Python/demoImages/known/dad.jpg')
faceLoc=fr.face_locations(dadFace)[0]
dadEncoding=fr.face_encodings(dadFace)[0]

mumFace=fr.load_image_file('C:/Users/User/Documents/Python/demoImages/known/Mum.jpg')
faceLoc=fr.face_locations(mumFace)[0]
mumEncoding=fr.face_encodings(mumFace)[0]

staiceyFace=fr.load_image_file('C:/Users/User/Documents/Python/demoImages/known/Staicey.jpg')
faceLoc=fr.face_locations(staiceyFace)[0]
staiceyEncoding=fr.face_encodings(staiceyFace)[0]

ianFace=fr.load_image_file('C:/Users/User/Documents/Python/demoImages/known/Ian.jpg')
faceLoc=fr.face_locations(ianFace)[0]
ianEncoding=fr.face_encodings(ianFace)[0]

bettyFace=fr.load_image_file('C:/Users/User/Documents/Python/demoImages/known/Phoebe.jpg')
faceLoc=fr.face_locations(bettyFace)[0]
bettyEncoding=fr.face_encodings(bettyFace)[0]

knownEncodings=[dadEncoding,mumEncoding,staiceyEncoding,ianEncoding,bettyEncoding]
names=['Dad','Mum','Staicey','Ian','Betty']
font=cv2.FONT_HERSHEY_SIMPLEX

while True:
    ignore, frame=cam.read()
    frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faceLocations=fr.face_locations(frameRGB)
    unknownEncodings=fr.face_encodings(frameRGB,faceLocations)

    for faceLocation,unknownEncoding in zip(faceLocations,unknownEncodings):
        top,right,bottom,left=faceLocation
        cv2.rectangle(frame,(left,top),(right,bottom),(255,0,0),2)
        name='Unknown'
        matches=fr.compare_faces(knownEncodings,unknownEncoding)
        if True in matches:
            matchIndex=matches.index(True)
            name=names[matchIndex]
        cv2.putText(frame,name,(left,top),font,.75,(0,0,255),2)
    cv2.flip(frame,1)
    cv2.imshow('My Window',frame)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

