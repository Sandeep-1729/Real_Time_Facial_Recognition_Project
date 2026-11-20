import cv2, os, re
root='data/identity'; os.makedirs(root,exist_ok=True)
name=input('Participant name: ').strip()
if not name or not re.fullmatch(r'[A-Za-z0-9 _-]+',name): raise SystemExit('Use letters, numbers, spaces, _ or -')
folder=os.path.join(root,name.replace(' ','_')); os.makedirs(folder,exist_ok=True)
cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
if not cap.isOpened(): raise SystemExit('Webcam unavailable; check Windows camera permissions.')
n=0; print('Look at camera, vary angles slightly. Q stops.')
while n<60:
 ok,frame=cap.read()
 if not ok: break
 gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 faces=cascade.detectMultiScale(gray,1.2,5,minSize=(80,80))
 for x,y,w,h in faces[:1]:
  face=cv2.resize(gray[y:y+h,x:x+w],(200,200)); cv2.imwrite(os.path.join(folder,f'{n:03}.jpg'),face); n+=1
  cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2); cv2.putText(frame,f'Saved {n}/60',(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
 cv2.imshow('Register face - Q to quit',frame)
 if cv2.waitKey(1)&255==ord('q'): break
cap.release(); cv2.destroyAllWindows(); print('Saved',n,'face crops in',folder)
