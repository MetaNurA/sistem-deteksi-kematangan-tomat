import cv2
import numpy as np

# Tentukan ukuran standar untuk semua gambar
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# Tentukan kernel untuk blurring
BLUR_KERNEL = (5, 5)

def clean_image(frame):
    """
    Fungsi ini menerima frame mentah dari kamera dan melakukan
    semua langkah preprocessing (Resize, Blur, BGR ke HSV).
    """
    # 1. Resize Gambar
    resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # 2. Noise Reduction (Blurring)
    blurred_frame = cv2.GaussianBlur(resized_frame, BLUR_KERNEL, 0)

    # 3. Konversi Color Space (BGR ke HSV)
    hsv_image = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    return hsv_image