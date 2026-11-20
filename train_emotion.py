import argparse, os, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers,models
p=argparse.ArgumentParser(); p.add_argument('--csv',default='data/fer2013.csv'); p.add_argument('--epochs',type=int,default=25); p.add_argument('--batch-size',type=int,default=64); a=p.parse_args()
if not os.path.isfile(a.csv): raise SystemExit('Put fer2013.csv in data/; see README.md')
df=pd.read_csv(a.csv); df.columns=[str(c).strip().lower() for c in df.columns]
if not {'emotion','pixels'}.issubset(df.columns): raise SystemExit("CSV needs 'emotion' and 'pixels' columns")
x=np.stack(df.pixels.astype(str).apply(lambda s:np.fromstring(s,sep=' ',dtype=np.uint8)).values)
if x.shape[1]!=2304: raise SystemExit('Each row must contain 2304 pixels (48x48).')
x=x.reshape(-1,48,48,1).astype('float32')/255.; y=df.emotion.astype('int64').to_numpy()
if 'usage' in df.columns:
 u=df.usage.astype(str).str.lower().to_numpy(); tr=u=='training'; va=u=='publictest'; te=u=='privatetest'
 if tr.any() and va.any(): X,Y=x[tr],y[tr]; VX,VY=x[va],y[va]; TX,TY=x[te],y[te]
 else: tr=va=te=None
else: tr=va=te=None
if tr is None:
 from sklearn.model_selection import train_test_split
 X,TX,Y,TY=train_test_split(x,y,test_size=.2,random_state=42,stratify=y); VX,TX,YV,TY=train_test_split(TX,TY,test_size=.5,random_state=42,stratify=TY)
else: YV=VY
m=models.Sequential([layers.Input((48,48,1)),layers.Conv2D(32,3,padding='same',activation='relu'),layers.BatchNormalization(),layers.MaxPooling2D(),layers.Dropout(.25),layers.Conv2D(64,3,padding='same',activation='relu'),layers.BatchNormalization(),layers.MaxPooling2D(),layers.Dropout(.25),layers.Conv2D(128,3,padding='same',activation='relu'),layers.BatchNormalization(),layers.MaxPooling2D(),layers.Dropout(.3),layers.Flatten(),layers.Dense(128,activation='relu'),layers.Dropout(.4),layers.Dense(7,activation='softmax')])
m.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']); os.makedirs('models',exist_ok=True)
cb=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True),tf.keras.callbacks.ModelCheckpoint('models/emotion_model.keras',monitor='val_accuracy',save_best_only=True)]
m.fit(X,Y,validation_data=(VX,YV),epochs=a.epochs,batch_size=a.batch_size,callbacks=cb); print('Test:',m.evaluate(TX,TY,verbose=0)); m.save('models/emotion_model.keras'); open('models/emotion_labels.txt','w').write('Angry\nDisgust\nFear\nHappy\nSad\nSurprise\nNeutral')
