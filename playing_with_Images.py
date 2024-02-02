import cv2 as cv
import sys
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

class KNN():
    def __init__(self , neighbours=5):
        self.neighbours=neighbours

    def fit(self,x_train , y_train):
        self.x=x_train.astype(np.int64)
        self.y=y_train

    def predict(self , x_test):
        self.x_point=x_test
        dist_list=[]
        for x,y in zip(self.x,self.y):
            dist=((x-self.x_point)**2).sum()
            dist_list.append([dist,y])
        dist_list=sorted(dist_list)
        top_vals=dist_list[:self.neighbours]

        items , count = np.unique(np.array(top_vals)[:,1] , return_counts=True)
        return items[np.argmax(count)]

model=KNN()

data=np.load("X:\ML\learning\data.npy")
x_train=data[ : , 1:]
y_train=data[ : ,0]

model.fit(x_train , y_train)

image=Image.open("X:\ML\sample.jpg")
img_arr=np.array(image)
#print(img_arr)
pos=img_arr.shape

new_img_arr=np.array([[0]*pos[1] for _ in range(pos[0])])
for i in range(pos[1]):
    for j in range(pos[0]):
        sum_arr=img_arr[i][j].sum()
        new_img_arr[i][j] = (sum_arr%255)

plt.imshow(new_img_arr)
print("number is: ", model.predict(new_img_arr.reshape(784)))

