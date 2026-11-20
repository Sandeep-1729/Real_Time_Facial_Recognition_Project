import cv2,os,json,numpy as np
EMOTIONS=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
for p in ['models/identity_lbph.yml','models/identity_labels.json','models/emotion_model.keras']:
 if not os.path.exists(p): raise SystemExit('Missing '+p+'; follow README training steps first.')
if not hasattr(cv2,'face'): raise SystemExit('Install opencv-contrib-python.')
import tensorflow as tf
emotion_model=tf.keras.models.load_model('models/emotion_model.keras'); rec=cv2.face.LBPHFaceRecognizer_create(); rec.read('models/identity_lbph.yml'); names=json.load(open('models/identity_labels.json'))
cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml'); cap=cv2.VideoCapture(0)
if not cap.isOpened(): raise SystemExit('Cannot open webcam. Check Windows camera access.')
print('Press Q to quit. LBPH threshold=75; tune with your own test captures.')
while True:
 ok,frame=cap.read()
 if not ok: break
 gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 for x,y,w,h in cascade.detectMultiScale(gray,1.2,5,minSize=(70,70)):
  roi=gray[y:y+h,x:x+w]; label,dist=rec.predict(cv2.resize(roi,(200,200)))
  who=names.get(str(label),'Unknown') if dist<75 else 'Unknown person'; color=(0,200,0) if who!='Unknown person' else (0,0,255)
  emo=emotion_model.predict(cv2.resize(roi,(48,48)).astype('float32').reshape(1,48,48,1)/255.,verbose=0)[0]; expression=EMOTIONS[int(np.argmax(emo))]
  cv2.rectangle(frame,(x,y),(x+w,y+h),color,2); cv2.putText(frame,who,(x,max(22,y-28)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2); cv2.putText(frame,'Expression: '+expression,(x,y+h+24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,220,0),2)
 cv2.imshow('Face identity + expression | Q to quit',frame)
 if cv2.waitKey(1)&255==ord('q'): break
cap.release(); cv2.destroyAllWindows()
