# Python code to read image
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

img = cv2.imread('concatenated_portion.png', cv2.IMREAD_GRAYSCALE)


cA, (cH, cV, cD) = pywt.dwt2(img, 'bior3.9')
print(pywt.wavelist())
print(np.shape(cA))
print(cA)
print("*****************************************************")
print(cH)



plt.subplot(2,2,1)
plt.imshow(np.uint16(cA),cmap='gray')
plt.show
plt.subplot(2,2,2)
plt.imshow(np.uint8(cH),cmap='gray')
plt.show
plt.subplot(2,2,3)
plt.imshow(np.uint8(cV),cmap='gray')
plt.subplot(2,2,4)
plt.imshow(np.uint8(cD),cmap='gray')
plt.show()



for i in range(521):
   for j in range(2449):
        cH[i,j] = 0
for i in range(519):
   for j in range(2449):
       cV[i,j] = 0
for i in range(518):
   for j in range(2446):
       cD[i,j] = 0
coeffs=cA,(cH,cV,cD)
new_img = pywt.idwt2(coeffs, 'bior3.9')
print(np.shape(cD))



plt.subplot(2,2,1)
plt.imshow(np.uint16(cA),cmap='gray')
plt.show
plt.subplot(2,2,2)
plt.imshow(np.uint8(cH),cmap='gray')
plt.show
plt.subplot(2,2,3)
plt.imshow(np.uint8(cV),cmap='gray')
plt.subplot(2,2,4)
plt.imshow(np.uint8(cD),cmap='gray')
plt.show()

plt.subplot(1,2,1)
plt.imshow(np.uint8(new_img),cmap='gray')
plt.show
plt.subplot(1,2,2)
plt.imshow(img,cmap='gray')
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
cv2.imwrite('new_img.png',new_img)
cv2.imwrite('four_new_imag.png',new_img_magnitude)