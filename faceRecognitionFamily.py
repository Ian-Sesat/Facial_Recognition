import face_recognition as fr
import cv2

#Training on our data and placing the trained data on our knownEncodings Array: 
staiceyFace=fr.load_image_file('C:/Users/User/Documents/Python/demoImages/known/Staicey.jpg')
faceLoc=fr.face_locations(staiceyFace)[0]
#print (faceLoc)
staiceyEncoding=fr.face_encodings(staiceyFace)[0]

ianFace=fr.load_image_file('C:/Users/User/Documents/Python/demoImages/known/Ian.jpg')
faceLoc=fr.face_locations(ianFace)[0]
ianEncoding=fr.face_encodings(ianFace)[0]

phoebeFace=fr.load_image_file('C:/Users/User/Documents/Python/demoImages/known/Phoebe.jpg')
faceLoc=fr.face_locations(phoebeFace)[0]
phoebeEncoding=fr.face_encodings(phoebeFace)[0]

mumFace=fr.load_image_file('C:/Users/User/Documents/Python/demoImages/known/Mum.jpg')
faceLoc=fr.face_locations(mumFace)[0]
mumEncoding=fr.face_encodings(mumFace)[0]

dadFace=fr.load_image_file('C:/Users/User/Documents/Python/demoImages/known/dad.jpg')
faceLoc=fr.face_locations(dadFace)[0]
dadEncoding=fr.face_encodings(dadFace)[0]

knownEncodings=[dadEncoding,mumEncoding,staiceyEncoding,ianEncoding,phoebeEncoding]
names=['Dad','Mum','Staicey','Ian','Phoebe']

unKnownFaces=fr.load_image_file('C:/Users/User/Documents/Python/demoImages/unknown/u14.jpg')
unknownFacesBGR=cv2.cvtColor(unKnownFaces,cv2.COLOR_RGB2BGR)
faceLocations=fr.face_locations(unKnownFaces)
#print(faceLocations)
unknownEncodings=fr.face_encodings(unKnownFaces,faceLocations)
font=cv2.FONT_HERSHEY_SIMPLEX

for faceLocation, unknownEncoding in zip(faceLocations,unknownEncodings):
    top,right,bottom,left=faceLocation
    cv2.rectangle(unknownFacesBGR,(left,top),(right,bottom),(255,0,0),2)
    name='Unknown'
    matches=fr.compare_faces(knownEncodings,unknownEncoding)
    #print(matches)
    if True in matches:
        matchIndex=matches.index(True)
        name=names[matchIndex]
        print(name)
    cv2.putText(unknownFacesBGR,name,(left,top),font,.75,(0,0,255),2)
cv2.resize(unknownFacesBGR,(640,360))
cv2.imshow('My Faces',unknownFacesBGR)
cv2.waitKey(5000)






