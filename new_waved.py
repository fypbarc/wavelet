import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate, upfirdn,normalize
from sklearn.preprocessing import minmax_scale

wavelet_sym4 = pywt.Wavelet('db4')
[phi_1, psi_1,x_1] = wavelet_sym4.wavefun(level=3)

wavelet_bior4 = pywt.Wavelet('sym4')
[phi_2, psi_2,x_2] = wavelet_bior4.wavefun(level=3)




print(np.shape(psi_1),np.shape(psi_2))

cust_phi = np.add(phi_2,phi_1)
cust_phi = cust_phi/5
cust_psi = np.add(psi_2,psi_1)
cust_psi = cust_psi/5

downsampled_cust_phi = decimate(cust_phi, q=2)
downsampled_cust_phi = minmax_scale(downsampled_cust_phi, feature_range=(-1,1))

downsampled_cust_psi = decimate(cust_psi,q=2)
downsampled_cust_psi = minmax_scale(downsampled_cust_psi, feature_range=(-1,1))
plt.stem( downsampled_cust_psi, use_line_collection = True)
plt.show()
upsampled_cust_phi = upfirdn([1, 1, 1],cust_phi,up=2)

hpf_decomp = downsampled_cust_phi[::-1]
plt.stem( hpf_decomp, use_line_collection = True)
plt.show()
lpf_recon = -1*hpf_decomp[::-1]
plt.stem( lpf_recon, use_line_collection = True)
plt.show()
hpf_recon = downsampled_cust_phi[::-1]



class cust_filterbank(object):
    @property
    def filter_bank(self):

        return [downsampled_cust_phi,hpf_decomp,lpf_recon,hpf_recon]
filter_bank = cust_filterbank()
myOtherWavelet = pywt.Wavelet(name="myHaarWavelet", filter_bank=filter_bank)

img = cv2.imread('concatenated_portion.png', cv2.IMREAD_GRAYSCALE)


cA, (cH, cV, cD),(cH1,cV1,cD1),(cH2,cV2,cD2) = pywt.wavedec2(img, myOtherWavelet,level=3)
print(pywt.wavelist())
print(np.shape(cA))

[r1,c1]=np.shape(cD)
[r2,c2]=np.shape(cD1)
[r3,c3]=np.shape(cD2)

cA_cV1, (cH_cV1, cV_cV1, cD_cV1) = pywt.wavedec2(cV1,myOtherWavelet,level=1)



plt.subplot(2,2,1)
plt.imshow(np.uint8(cA_cV1),cmap='gray')
plt.title('cA')
plt.show
plt.subplot(2,2,2)
plt.imshow(np.uint8(cH_cV1),cmap='gray')
plt.title('cH')
plt.show
plt.subplot(2,2,3)
plt.imshow(np.uint8(cV_cV1),cmap='gray')
plt.title('cV')
plt.show
plt.subplot(2,2,4)
plt.imshow(np.uint8(cD_cV1),cmap='gray')
plt.title('cD')
plt.show()




plt.subplot(5,2,1)
plt.imshow(np.uint8(cA),cmap='gray')
plt.title('cA')
plt.show
plt.subplot(5,2,2)
plt.imshow(np.uint8(cH),cmap='gray')
plt.title('cH')
plt.show
plt.subplot(5,2,3)
plt.imshow(np.uint8(cV),cmap='gray')
plt.title('cV')
plt.show
plt.subplot(5,2,4)
plt.imshow(np.uint8(cD),cmap='gray')
plt.title('cD')
plt.show
plt.subplot(5,2,5)
plt.imshow(np.uint8(cH1),cmap='gray')
plt.title('cH1')
plt.show
plt.subplot(5,2,6)
plt.imshow(np.uint8(cV1),cmap='gray')
plt.title('cV1')
plt.show
plt.subplot(5,2,7)
plt.imshow(np.uint8(cD1),cmap='gray')
plt.title('cD1')
plt.subplot(5,2,8)
plt.imshow(np.uint8(cH2),cmap='gray')
plt.title('cH2')
plt.show
plt.subplot(5,2,9)
plt.imshow(np.uint8(cV2),cmap='gray')
plt.title('cV2')
plt.show
plt.subplot(5,2,10)
plt.imshow(np.uint8(cD2),cmap='gray')
plt.title('cD2')
plt.show()


[r6,c6] = np.shape(cV_cV1)



for i in range(r6):
    for j in range(c6):
        if cD_cV1[i,j]>200:
            cA_cV1[i,j] = 0
            cH_cV1[i, j] = 0
for i in range(r2):
    for j in range(c2):
        cD1[i,j] = 0

for i in range(r3):
    for j in range(c3):
        cV2[i,j] = 0
        cD2[i,j] = 0



coeffs_cV1 = cA_cV1, (cH_cV1, cV_cV1, cD_cV1)
new_cV1= pywt.waverec2(coeffs_cV1, myOtherWavelet)

[r5,c5] = np.shape(new_cV1)

new_cV1 = np.delete(new_cV1,273,0)
new_cV1 = np.delete(new_cV1,1237,1)


print(np.shape(new_cV1))
print(np.shape(cV1))


coeffs=cA,(cH,cV,cD),(cH1,new_cV1,cD1),(cH2,cV2,cD2)
new_img = pywt.waverec2(coeffs, myOtherWavelet)
print(np.shape(img))
print(np.shape(new_img))

plt.subplot(1,2,1)
plt.imshow(np.uint8(img),cmap='gray')
plt.title('Input image')
plt.show
plt.subplot(1,2,2)
plt.imshow(np.uint8(new_img),cmap='gray')
plt.title('Output image')
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
plt.title("Input Image FFT")
plt.show
plt.subplot(1,2,2)
plt.imshow(np.uint8(new_img_magnitude),cmap='gray')
plt.title('Ouput Image FFT')
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