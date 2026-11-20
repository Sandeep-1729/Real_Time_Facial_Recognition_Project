import cv2, os, json, numpy as np
if not hasattr(cv2,'face'): raise SystemExit('Install opencv-contrib-python in this environment.')
base='data/identity'; images=[]; labels=[]; names={}
for i,name in enumerate(sorted(os.listdir(base))):
 folder=os.path.join(base,name)
 if not os.path.isdir(folder): continue
 for fn in os.listdir(folder):
  im=cv2.imread(os.path.join(folder,fn),0)
  if im is not None: images.append(cv2.resize(im,(200,200))); labels.append(i)
 if any(os.path.isfile(os.path.join(folder,f)) and f.lower().endswith(('.jpg','.png','.jpeg')) for f in os.listdir(folder)): names[str(i)]=name.replace('_',' ')
if len(images)<2: raise SystemExit('Register face images first: python register_face.py')
os.makedirs('models',exist_ok=True); rec=cv2.face.LBPHFaceRecognizer_create(); rec.train(images,np.array(labels,dtype=np.int32)); rec.save('models/identity_lbph.yml')
json.dump(names,open('models/identity_labels.json','w'),indent=2); print('Identity model trained on',len(images),'images.')
