import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

img = cv2.imread('concatenated.png', cv2.IMREAD_GRAYSCALE)


cA, (cH, cV, cD),(cH1,cV1,cD1),(cH2,cV2,cD2),(cH3,cV3,cD3) = pywt.wavedec2(img, 'bior3.9',level=4)
print(pywt.wavelist())
print(np.shape(cA))

[r1,c1]=np.shape(cD)
[r2,c2]=np.shape(cD1)
[r3,c3]=np.shape(cD2)
[r4,c4]=np.shape(cD3)


plt.subplot(10,2,1)
plt.imshow(np.uint8(cA),cmap='gray')
plt.show
plt.subplot(10,2,2)
plt.imshow(np.uint8(cH),cmap='gray')
plt.show
plt.subplot(10,2,3)
plt.imshow(np.uint8(cV),cmap='gray')
plt.show
plt.subplot(10,2,4)
plt.imshow(np.uint8(cD),cmap='gray')
plt.show
plt.subplot(10,2,5)
plt.imshow(np.uint8(cH1),cmap='gray')
plt.show
plt.subplot(10,2,6)
plt.imshow(np.uint8(cV1),cmap='gray')
plt.show
plt.subplot(10,2,7)
plt.imshow(np.uint8(cD1),cmap='gray')
plt.subplot(10,2,8)
plt.imshow(np.uint8(cH2),cmap='gray')
plt.show
plt.subplot(10,2,9)
plt.imshow(np.uint8(cV2),cmap='gray')
plt.show
plt.subplot(10,2,10)
plt.imshow(np.uint8(cD2),cmap='gray')
plt.subplot(10,2,11)
plt.imshow(np.uint8(cH3),cmap='gray')
plt.show
plt.subplot(10,2,12)
plt.imshow(np.uint8(cV3),cmap='gray')
plt.show
plt.subplot(10,2,13)
plt.imshow(np.uint8(cD3),cmap='gray')
plt.show()

for i in range(r2):
    for j in range(c2):
        cD1[i,j] = 0
for i in range(r2):
    for j in range(c2):
        cV1[i,j] = 0
for i in range(r3):
    for j in range(c3):
        cD2[i,j] = 0
for i in range(r4):
    for j in range(c4):
        cH3[i,j] = 0
for i in range(r4):
    for j in range(c4):
        cD3[i,j] = 0
for i in range(r4):
    for j in range(c4):
        cV3[i,j] = 0
coeffs=cA,(cH,cV,cD),(cH1,cV1,cD1),(cH2,cV2,cD2),(cH3,cV3,cD3)
new_img = pywt.waverec2(coeffs, 'bior3.9')

plt.subplot(1,2,1)
plt.imshow(np.uint8(new_img),cmap='gray')
plt.show
plt.subplot(1,2,2)
plt.imshow(img,cmap='gray')
plt.show()

img_fourier = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
img_fourier_shift = np.fft.fftshift(img_fourier)
img_magnitude = 20 * np.log(cv2.magnitude(img_fourier_shift[:, :, 0], img_fourier_shift[:, :, 1]))
img_magnitude = cv2.normalize(img_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)


new_img_fourier = cv2.dft(np.float32(new_img), flags=cv2.DFT_COMPLEX_OUTPUT)
new_img_fourier_shift = np.fft.fftshift(new_img_fourier)
new_img_magnitude = 20 * np.log(cv2.magnitude(new_img_fourier_shift[:, :, 0], new_img_fourier_shift[:, :, 1]))
new_img_magnitude = cv2.normalize(new_img_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

plt.subplot(1,2,1)
plt.imshow(np.uint8(img_magnitude),cmap='gray')
plt.show
plt.subplot(1,2,2)
plt.imshow(np.uint8(new_img_magnitude),cmap='gray')
plt.show()

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * (np.log10(max_pixel / np.sqrt(mse)))
    return psnr
value = PSNR(img, new_img)
print(f"PSNR value is {value} dB")