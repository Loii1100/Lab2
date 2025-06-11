from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math # Cần cho log transformation

def load_and_convert_to_gray(image_path):
    """Loads an image and converts it to grayscale."""
    img = Image.open(image_path).convert('L') # 'L' for grayscale
    return img, np.asarray(img)

def image_inverse_transformation(image_path):
    """
    1.1. Thực hiện biến đổi ảnh nghịch đảo (Inverse Transformation).
    """
    img, im_1 = load_and_convert_to_gray(image_path)

    # Inversion operation
    im_2 = 255 - im_1
    new_img = Image.fromarray(im_2)

    print(f"Áp dụng Biến đổi nghịch đảo cho {image_path}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')
    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title('Ảnh Nghịch đảo')
    plt.show()

def gamma_correction(image_path, gamma=0.5):
    """
    1.2. Thay đổi chất lượng ảnh với Power law (Gamma-Correction).
    """
    img, im_1 = load_and_convert_to_gray(image_path)
    b1 = im_1.astype(float)
    b2 = np.max(b1) # Maximum pixel value

    # Normalize pixels to range [0, 1]
    normalized_pixels = b1 / b2

    # Apply gamma correction formula
    gamma_corrected_pixels = np.power(normalized_pixels, gamma) * 255.0

    c1 = gamma_corrected_pixels.astype(np.uint8)
    new_img = Image.fromarray(c1)

    print(f"Áp dụng Gamma Correction (gamma={gamma}) cho {image_path}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')
    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title(f'Ảnh sau Gamma Correction (γ={gamma})')
    plt.show()

def log_transformation(image_path):
    """
    1.3. Thay đổi cường độ điểm ảnh với Log Transformation.
    """
    img, im_1 = load_and_convert_to_gray(image_path)
    b1 = im_1.astype(float)
    b2 = np.max(b1) # Maximum pixel value

    # c = L-1 / log(1+Lmax)
    c = 255 / np.log(1 + b2)

    # Apply log transformation
    log_transformed_pixels = c * np.log(1 + b1)

    c1 = log_transformed_pixels.astype(np.uint8)
    new_img = Image.fromarray(c1)

    print(f"Áp dụng Log Transformation cho {image_path}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')
    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title('Ảnh sau Log Transformation')
    plt.show()

def histogram_equalization(image_path):
    """
    1.4. Cân bằng biểu đồ (Histogram equalization).
    """
    img, iml = load_and_convert_to_gray(image_path)
    b1 = iml.flatten() # Convert 2D array to 1D array

    # Compute histogram
    hist, bins = np.histogram(b1, 256, [0, 255])

    # Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Mask places where cdf=0 to avoid division by zero
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    # Normalize the CDF
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    
    # Fill masked values with 0
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Apply equalization to the flattened image
    im2 = cdf[b1]
    
    # Reshape back to 2D
    im3 = np.reshape(im2, iml.shape)
    new_img = Image.fromarray(im3)

    print(f"Áp dụng Cân bằng biểu đồ (Histogram Equalization) cho {image_path}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')
    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title('Ảnh sau Cân bằng biểu đồ')
    plt.show()

def contrast_stretching(image_path):
    """
    1.5. Thay đổi ảnh với Contrast Stretching.
    """
    img, iml = load_and_convert_to_gray(image_path)

    a = iml.min() # min pixel value
    b = iml.max() # max pixel value

    # Avoid division by zero if all pixels are the same
    if (b - a) == 0:
        print("Không thể thực hiện Contrast Stretching: Tất cả pixel có cùng giá trị.")
        plt.imshow(img, cmap='gray')
        plt.title('Ảnh Gốc (không thay đổi)')
        plt.show()
        return

    # Apply contrast stretching formula
    # out = (in - a) * (255 / (b - a))
    im2 = ((iml - a) * (255.0 / (b - a))).astype(np.uint8)

    new_img = Image.fromarray(im2)

    print(f"Áp dụng Contrast Stretching cho {image_path}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')
    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title('Ảnh sau Contrast Stretching')
    plt.show()

# (Thêm các hàm cho FFT, Butterworth Lowpass, Butterworth Highpass vào filters.py)
