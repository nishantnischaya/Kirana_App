import os
import cv2
import numpy as  np
from tqdm import tqdm

REBUILD_DATA =True
folder=r"C:\Users\prath\Desktop\New folder (2)\Train\\"
class datahandling():
    img1=120
    img2=160
    bingo=folder+"bingo"
    colgate=folder+"colgate"
    cs=folder+"cs"
    #l52g=folder+"l52g"
    #l90g=folder+"l90g"
    ms=folder+"ms"
    ml=folder+"ml"
    labels={bingo:0,colgate:1,cs:2,ms:3,ml:4}
    tdata=[]
    bc=0
    cgc=0
    csc=0
   # l52gc=0
    #l90gc=0
    msc=0
    mlc=0
    def mtraind(self):
        for label in self.labels:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path=os.path.join(label,f)
                    img=cv2.imread(path)
                    print(img.shape)
                    img[:,:,1]=np.zeros([img.shape[0],img.shape[1]])
                    
                    img=cv2.resize(img,(self.img2,self.img1))
                    print(img.shape)
                    self.tdata.append([np.array(img),np.eye(5)[self.labels[label]]])
                    if label==self.bingo:
                        self.bc+=1
                    elif label==self.colgate:
                        self.cgc+=1
                    elif label==self.cs:
                        self.csc+=1
 #                   elif label==self.l52g:
  #                      self.l52gc+=1
  #                  elif label==self.l90g:
   #                     self.l90gc+=1
                    elif label==self.ms:
                        self.msc+=1
                    elif label==self.ml:
                        self.mlc+=1
                except exception as e:
                    pass
        np.random.shuffle(self.tdata)
        np.save("traindata.npy",self.tdata)
        print("bingo",self.bc)
        print("colgate",self.cgc)
        print("colgatesalt",self.csc)
    #    print("lays50g",self.l52gc)
     #   print("lays90g",self.l90gc)
        print("maggis",self.msc)
        print("maggil",self.mlc)
if REBUILD_DATA:
    d=datahandling()
    d.mtraind()
