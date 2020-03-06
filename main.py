import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from utils import *
import pickle
images_path = '.images/'

im = cv2.imread('tordo.JPG')

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


i, j = 1000, 500
rect = patches.Rectangle((i, j), 2048, 2048, linewidth=1, edgecolor='r', facecolor='none')
fig, ax = plt.subplots(1, dpi=300)
ax.imshow(gray, cmap='gray')
ax.add_patch(rect)
plt.show()


X = gray[j:j + 2048, i:i + 2048]
plt.figure(dpi=300)
plt.imshow(X, cmap='gray')
plt.savefig(images_path+'tordo_cut.png', dpi=300)
plt.show()
Tordo = HaarFilterBank(X)
Tordo.get_IDWT()
Tordo.makeAllReconstructions()


# book
im = cv2.imread('book.JPG')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
i, j = 2000, 500
rect = patches.Rectangle((i, j), 2048, 2048, linewidth=1, edgecolor='r', facecolor='none')
fig, ax = plt.subplots(1, dpi=300)
ax.imshow(gray, cmap='gray')
ax.add_patch(rect)
plt.show()


X2 = gray[j:j + 2048, i:i + 2048]
plt.figure(dpi=300)
plt.imshow(X2, cmap='gray')
plt.savefig(images_path+'book_cut.png', dpi=300)
plt.show()

Book = HaarFilterBank(X2)
Book.get_IDWT()
Book.makeAllReconstructions()


# saving objects
with open('Tordo.Haar', 'wb') as f:
    pickle.dump(Tordo, f)
f.close()

with open('Book.Haar', 'wb') as f:
    pickle.dump(Book, f)
f.close()
