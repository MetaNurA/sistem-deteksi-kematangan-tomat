import cv2
import numpy as np
import os

def preprocess(image_path, output_dir="hasil_preprocess"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)

    if image is None:
        print(f"[GAGAL] Tidak bisa membaca gambar: {image_path}")
        return

    # 1️⃣ Konversi ke HSV & hilangkan noise
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)

    # 2️⃣ Range warna HSV untuk tiga tingkat kematangan
    ranges = {
        "Hijau": ([35, 50, 50], [85, 255, 255]),
        "Oranye": ([10, 100, 100], [25, 255, 255]),
        "Merah": ([0, 100, 100], [10, 255, 255])
    }

    kernel = np.ones((5, 5), np.uint8)
    masks = []
    hasil_persentase = {}

    # 3️⃣ Proses tiap warna
    for warna, (lower, upper) in ranges.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask = cv2.inRange(blurred, lower_np, upper_np)

        # Bersihkan mask biar halus
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Hitung persentase area putih (warna terdeteksi)
        white_pixels = cv2.countNonZero(mask_clean)
        total_pixels = mask_clean.size
        percentage = (white_pixels / total_pixels) * 100
        hasil_persentase[warna] = percentage

        # Ubah ke 3 channel biar bisa digabung
        mask_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)

        # Tambahkan teks label
        label = np.zeros_like(mask_bgr)
        cv2.putText(label, f"{warna} ({percentage:.1f}%)", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        labeled_mask = cv2.addWeighted(mask_bgr, 1, label, 0.6, 0)
        masks.append(labeled_mask)

    # 4️⃣ Gabungkan citra asli + mask ke satu frame
    image_label = image.copy()
    cv2.putText(image_label, "Citra Asli", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    image_resized = cv2.resize(image_label, (300, 300))
    masks_resized = [cv2.resize(m, (300, 300)) for m in masks]
    row1 = np.hstack([image_resized])
    row2 = np.hstack(masks_resized)
    combined = np.vstack([row1, row2])

    # 5️⃣ Tentukan warna dominan
    warna_dominan = max(hasil_persentase, key=hasil_persentase.get)
    cv2.putText(combined, f"Dominan: {warna_dominan}",
                (20, 590), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 6️⃣ Simpan hasil gabungan
    output_path = f"{output_dir}/{base_name}_gabungan.jpg"
    cv2.imwrite(output_path, combined)

    print(f"\n[HASIL] {base_name}:")
    for warna, persen in hasil_persentase.items():
        print(f"  - {warna}: {persen:.2f}% area terdeteksi")
    print(f"  => Dominan: {warna_dominan}")
    print(f"[OK] Gambar disimpan di: {output_path}\n")

    # 7️⃣ Tampilkan hasil
    cv2.imshow(f"Hasil {base_name}", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ======== Jalankan untuk 3 gambar ========
gambar_tomat = [
    "Assets/tomat_hijau.jpg",
    "Assets/tomat_oranye.jpg",
    "Assets/tomat_merah.jpg"
]

for img in gambar_tomat:
    preprocess(img)
