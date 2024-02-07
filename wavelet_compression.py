import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
import os

img_og = cv2.imread('tp.png', cv2.IMREAD_GRAYSCALE)

cA, (cH, cV, cD),(cH1,cV1,cD1),(cH2,cV2,cD2),(cH3,cV3,cD3) = pywt.wavedec2(img_og, 'bior3.9',level=4)

coeffs=[cA,(cH,cV,cD),(cH1,cV1,cD1),(cH2,cV2,cD2),(cH3,cV3,cD3)]

for image in coeffs:
    if not isinstance(image,tuple):
        for i in range(np.shape(image)[0]):
            for j in range(np.shape(image)[1]):
                image[i,j] = (image[i, j])*0.01
    elif isinstance(image,tuple):
        for img in image:
            for i in range(np.shape(img)[0]):
                for j in range(np.shape(img)[1]):
                    img[i, j] = img[i, j] * 0.01

coeffs=cA,(cH,cV,cD),(cH1,cV1,cD1),(cH2,cV2,cD2),(cH3,cV3,cD3)
new_img = pywt.waverec2(coeffs, 'bior3.9')

cv2.imwrite('compressed.png', np.uint8(new_img))


img_read = cv2.imread('compressed.png', cv2.IMREAD_GRAYSCALE)

_cA, (_cH, _cV, _cD),(_cH1,_cV1,_cD1),(_cH2,_cV2,_cD2),(_cH3,_cV3,_cD3) = pywt.wavedec2(img_read, 'bior3.9',level=4)

_coeffs=[_cA, (_cH, _cV, _cD),(_cH1,_cV1,_cD1),(_cH2,_cV2,_cD2),(_cH3,_cV3,_cD3)]

for image in _coeffs:
    if not isinstance(image,tuple):
        for i in range(np.shape(image)[0]):
            for j in range(np.shape(image)[1]):
                image[i,j] = (image[i, j])*(1/0.01)


    elif isinstance(image,tuple):
        for img in image:
            for i in range(np.shape(img)[0]):
                for j in range(np.shape(img)[1]):
                    img[i, j] = img[i, j] * (1/0.01)


_new_img = pywt.waverec2(_coeffs, 'bior3.9')

plt.subplot(2,2,1)
plt.imshow((img_og),cmap='gray')
plt.show
plt.subplot(2,2,2)
plt.imshow((new_img),cmap='gray')
plt.show
plt.subplot(2,2,3)
plt.imshow((img_read),cmap='gray')
plt.show
plt.subplot(2,2,4)
plt.imshow((_new_img),cmap='gray')
plt.show()


