from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift, ifft2

def load_and_convert_to_gray(image_path):
    """Loads an image and converts it to grayscale."""
    img = Image.open(image_path).convert('L') # 'L' for grayscale
    return img, np.asarray(img)

def fast_fourier_transform(image_path):
    """
    1.6.1. Biến đổi ảnh với Fast Fourier Transform (FFT).
    """
    img, iml = load_and_convert_to_gray(image_path)

    # Perform FFT
    f = fft2(iml)
    
    # Shift the zero-frequency component to the center of the spectrum
    fshift = fftshift(f)
    
    # Compute the magnitude spectrum for visualization
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10) # Add small epsilon to avoid log(0)

    print(f"Áp dụng Fast Fourier Transform (FFT) cho {image_path}")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Phổ Biên độ FFT')
    plt.show()

def butterworth_lowpass_filter(image_path, D0=30.0, n=1):
    """
    1.6.2. Lọc trong miền tần số - Butterworth Lowpass Filter.
    D0: Cut-off radius
    n: Order of the filter
    """
    img, iml = load_and_convert_to_gray(image_path)

    # Perform FFT
    f = fft2(iml)
    fshift = fftshift(f)

    M, N = iml.shape
    H = np.zeros((M, N), dtype=np.float32)

    center1 = M / 2
    center2 = N / 2

    # Define the convolution function for BLPF
    for i in range(M):
        for j in range(N):
            r = np.sqrt((i - center1)**2 + (j - center2)**2)
            H[i, j] = 1 / (1 + (r / D0)**(2 * n))
    
    # Apply the filter in the frequency domain
    filtered_fshift = fshift * H

    # Inverse FFT and magnitude
    f_ishift = ifftshift(filtered_fshift)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize to 0-255 and convert to uint8
    img_back = np.uint8(255 * (img_back / np.max(img_back)))

    new_img = Image.fromarray(img_back)

    print(f"Áp dụng Butterworth Lowpass Filter (D0={D0}, n={n}) cho {image_path}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')
    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title('Ảnh sau Lọc Lowpass')
    plt.show()

def butterworth_highpass_filter(image_path, D0=30.0, n=1):
    """
    1.6.2. Lọc trong miền tần số - Butterworth Highpass Filter.
    D0: Cut-off radius
    n: Order of the filter
    """
    img, iml = load_and_convert_to_gray(image_path)

    # Perform FFT
    f = fft2(iml)
    fshift = fftshift(f)

    M, N = iml.shape
    H = np.zeros((M, N), dtype=np.float32)

    center1 = M / 2
    center2 = N / 2

    # Define the convolution function for BHPF
    for i in range(M):
        for j in range(N):
            r = np.sqrt((i - center1)**2 + (j - center2)**2)
            if r == 0: # Avoid division by zero at the center
                H[i, j] = 0
            else:
                H[i, j] = 1 / (1 + (D0 / r)**(2 * n))
    
    # Apply the filter in the frequency domain
    filtered_fshift = fshift * H

    # Inverse FFT and magnitude
    f_ishift = ifftshift(filtered_fshift)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Normalize to 0-255 and convert to uint8
    img_back = np.uint8(255 * (img_back / np.max(img_back)))

    new_img = Image.fromarray(img_back)

    print(f"Áp dụng Butterworth Highpass Filter (D0={D0}, n={n}) cho {image_path}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')
    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title('Ảnh sau Lọc Highpass')
    plt.show()

# You might need to import ifftshift explicitly if not automatically available with fftpack
from scipy.fftpack import ifftshift
