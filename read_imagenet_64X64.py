import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict
def load_databatch( img_size=64):

    d = unpickle('val_data')
    x = d['data']
    y = d['labels']

    x = x/np.float32(255)

    y = [i-1 for i in y]
    data_size = x.shape[0]

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    
    return X_train,Y_train
    
X_train,Y_train=load_databatch()
X_train=X_train.transpose(0,2,3,1)

# print(X_train.shape)
# print(len(Y_train))

x1=X_train[20,:,:,:]
y1=Y_train[20]

with open(r'C:\My_Data\trasnformer_code\imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

plt.figure()
plt.imshow(x1)
print(labels[y1])
