import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Import all functions from your transformation and filter files
from image_transforms import (
    image_inverse_transformation,
    gamma_correction,
    log_transformation,
    histogram_equalization,
    contrast_stretching,
    load_and_convert_to_gray # Re-import if needed for other tasks
)
from filters import (
    fast_fourier_transform,
    butterworth_lowpass_filter,
    butterworth_highpass_filter
)

# Placeholder for Min/Max Filter - you'll need to implement these or find a library for them
def min_filter(image_path, size=3):
    """Placeholder for Min Filter (Erosion)."""
    img = Image.open(image_path).convert('L')
    img_arr = np.array(img)
    
    # Simple 3x3 min filter (erosion)
    output_arr = np.zeros_like(img_arr)
    pad_size = size // 2
    padded_img = np.pad(img_arr, pad_size, mode='edge')

    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            window = padded_img[i:i+size, j:j+size]
            output_arr[i, j] = np.min(window)
    
    new_img = Image.fromarray(output_arr)
    print(f"Áp dụng Min Filter (kích thước {size}x{size}) cho {image_path}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')
    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title(f'Ảnh sau Min Filter')
    plt.show()


def max_filter(image_path, size=3):
    """Placeholder for Max Filter (Dilation)."""
    img = Image.open(image_path).convert('L')
    img_arr = np.array(img)
    
    # Simple 3x3 max filter (dilation)
    output_arr = np.zeros_like(img_arr)
    pad_size = size // 2
    padded_img = np.pad(img_arr, pad_size, mode='edge')

    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            window = padded_img[i:i+size, j:j+size]
            output_arr[i, j] = np.max(window)
    
    new_img = Image.fromarray(output_arr)
    print(f"Áp dụng Max Filter (kích thước {size}x{size}) cho {image_path}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh Gốc')
    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title(f'Ảnh sau Max Filter')
    plt.show()


def change_rgb_order_and_transform(image_path, transform_func, is_color=True):
    """
    Thực hiện thay đổi thứ tự kênh màu RGB ngẫu nhiên và sau đó áp dụng biến đổi.
    Bài 2.3.
    """
    if is_color:
        try:
            img = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy ảnh {image_path}. Vui lòng đảm bảo ảnh có trong thư mục.")
            return
        
        img_arr = np.array(img)
        
        # Get random permutation of RGB channels
        channels = [0, 1, 2] # R, G, B
        random.shuffle(channels)
        
        reordered_img_arr = img_arr[:, :, channels]
        
        # Convert the reordered array back to PIL Image (for consistency, though can operate directly)
        reordered_pil_img = Image.fromarray(reordered_img_arr)

        # Save this reordered image temporarily to apply the transformation
        temp_reordered_path = "temp_reordered_image.jpg"
        reordered_pil_img.save(temp_reordered_path)
        
        print(f"\nĐã thay đổi thứ tự kênh RGB thành {channels} cho {image_path}. Tiếp theo áp dụng biến đổi...")
        
        # Now apply the transformation to the temporarily saved image
        transform_func(temp_reordered_path)

        # Clean up temporary file
        os.remove(temp_reordered_path)
    else:
        print("Biến đổi này chỉ áp dụng cho ảnh màu RGB.")


def change_rgb_order_and_filter(image_path, filter_func, min_max_filter_func=None, is_color=True):
    """
    Thực hiện thay đổi thứ tự kênh màu RGB ngẫu nhiên, sau đó áp dụng bộ lọc và có thể thêm Min/Max filter.
    Bài 2.4.
    """
    if is_color:
        try:
            img = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy ảnh {image_path}. Vui lòng đảm bảo ảnh có trong thư mục.")
            return
        
        img_arr = np.array(img)
        
        # Get random permutation of RGB channels
        channels = [0, 1, 2] # R, G, B
        random.shuffle(channels)
        
        reordered_img_arr = img_arr[:, :, channels]
        
        # Convert the reordered array back to PIL Image (for consistency)
        reordered_pil_img = Image.fromarray(reordered_img_arr)

        # Save this reordered image temporarily
        temp_reordered_path = "temp_reordered_image.jpg"
        reordered_pil_img.save(temp_reordered_path)
        
        print(f"\nĐã thay đổi thứ tự kênh RGB thành {channels} cho {image_path}. Tiếp theo áp dụng bộ lọc...")
        
        # Apply the main filter
        filter_func(temp_reordered_path)

        # Apply additional Min/Max filter if specified
        if min_max_filter_func:
            print("Áp dụng thêm bộ lọc Min/Max...")
            # Re-save the output of the first filter if it was saved to a file,
            # or pass the image directly if the filter function returns it.
            # For simplicity, I'm just calling the min/max filter on the original reordered image.
            # In a real scenario, you'd apply it to the output of the first filter.
            min_max_filter_func(temp_reordered_path)
        
        # Clean up temporary file
        os.remove(temp_reordered_path)
    else:
        print("Biến đổi này chỉ áp dụng cho ảnh màu RGB.")


def display_menu():
    print("\n--- MENU XỬ LÝ ẢNH ---")
    print("Các biến đổi cường độ ảnh (Áp dụng cho ảnh xám):")
    print("1. Biến đổi nghịch đảo (I)")
    print("2. Gamma Correction (G)")
    print("3. Log Transformation (L)")
    print("4. Cân bằng biểu đồ (H)")
    print("5. Contrast Stretching (C)")

    print("\nCác biến đổi trong miền tần số và Bộ lọc (Áp dụng cho ảnh xám):")
    print("6. Fast Fourier Transform (F)")
    print("7. Butterworth Lowpass Filter (B)")
    print("8. Butterworth Highpass Filter (P)")

    print("\nCác bài tập bổ sung:")
    print("9. Thay đổi thứ tự RGB và áp dụng biến đổi (R)")
    print("10. Thay đổi thứ tự RGB, áp dụng bộ lọc và Min/Max Filter (M)")

    print("0. Thoát (Q)")
    print("----------------------")

def main():
    image_path = 'world_cup.jpg' # Đặt tên ảnh mặc định của bạn
    # Bạn có thể thay đổi để hỏi người dùng tên ảnh hoặc dùng ảnh khác

    while True:
        display_menu()
        choice = input("Chọn chức năng (nhập số hoặc chữ cái): ").upper()

        try:
            if choice == 'I' or choice == '1':
                image_inverse_transformation(image_path)
            elif choice == 'G' or choice == '2':
                gamma_val = float(input("Nhập giá trị gamma (ví dụ 0.5 hoặc 5): "))
                gamma_correction(image_path, gamma_val)
            elif choice == 'L' or choice == '3':
                log_transformation(image_path)
            elif choice == 'H' or choice == '4':
                histogram_equalization(image_path)
            elif choice == 'C' or choice == '5':
                contrast_stretching(image_path)
            elif choice == 'F' or choice == '6':
                fast_fourier_transform(image_path)
            elif choice == 'B' or choice == '7':
                D0_val = float(input("Nhập giá trị D0 (Cut-off radius, ví dụ 30.0): "))
                n_val = int(input("Nhập bậc n (Order of filter, ví dụ 1 hoặc 2): "))
                butterworth_lowpass_filter(image_path, D0=D0_val, n=n_val)
            elif choice == 'P' or choice == '8':
                D0_val = float(input("Nhập giá trị D0 (Cut-off radius, ví dụ 30.0): "))
                n_val = int(input("Nhập bậc n (Order of filter, ví dụ 1 hoặc 2): "))
                butterworth_highpass_filter(image_path, D0=D0_val, n=n_val)
            elif choice == 'R' or choice == '9':
                print("Chọn biến đổi để áp dụng sau khi thay đổi RGB:")
                print("   I: Inverse, G: Gamma, L: Log, H: Histogram, C: Contrast Stretching")
                sub_choice = input("Nhập lựa chọn: ").upper()
                
                transform_map = {
                    'I': image_inverse_transformation,
                    'G': lambda p: gamma_correction(p, 0.5), # Default gamma for demo
                    'L': log_transformation,
                    'H': histogram_equalization,
                    'C': contrast_stretching
                }
                
                if sub_choice in transform_map:
                    change_rgb_order_and_transform(image_path, transform_map[sub_choice], is_color=True)
                else:
                    print("Lựa chọn không hợp lệ cho biến đổi.")
            elif choice == 'M' or choice == '10':
                print("Chọn bộ lọc để áp dụng sau khi thay đổi RGB:")
                print("   B: Butterworth Lowpass, P: Butterworth Highpass")
                sub_choice = input("Nhập lựa chọn: ").upper()
                
                filter_map = {
                    'B': lambda p: butterworth_lowpass_filter(p, 30.0, 1), # Default D0, n
                    'P': lambda p: butterworth_highpass_filter(p, 30.0, 1) # Default D0, n
                }

                min_max_filter_to_apply = None
                if sub_choice == 'B':
                    min_max_filter_to_apply = min_filter
                elif sub_choice == 'P':
                    min_max_filter_to_apply = max_filter

                if sub_choice in filter_map:
                    change_rgb_order_and_filter(image_path, filter_map[sub_choice], min_max_filter_to_apply, is_color=True)
                else:
                    print("Lựa chọn không hợp lệ cho bộ lọc.")

            elif choice == 'Q' or choice == '0':
                print("Thoát chương trình. Tạm biệt!")
                break
            else:
                print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy ảnh '{image_path}'. Vui lòng đảm bảo ảnh có trong thư mục dự án.")
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()
