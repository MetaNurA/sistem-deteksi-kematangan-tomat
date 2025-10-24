## 1. Pemilihan Color Space: HSV
Kita akan menggunakan HSV (Hue, Saturation, Value).
- **Alasan Krusial:** Channel **H (Hue)** secara efektif memisahkan informasi warna murni dari **V (Value/Kecerahan)**. Ini membuat deteksi warna (seperti kematangan buah) lebih stabil dan robust terhadap perubahan intensitas cahaya (misalnya, perbedaan antara di bawah bayangan dan di bawah sinar matahari terang).

## 2. Pemilihan Noise Reduction: Gaussian Blur
Kita akan menggunakan `cv2.GaussianBlur` terlebih dahulu.
- **Alasan:** Untuk menghaluskan gambar dan menghilangkan *noise* frekuensi tinggi (bintik-bintik halus) dari sensor kamera. Ini membantu menyederhanakan gambar sebelum Anggota 3 melakukan segmentasi.
- **Catatan Alternatif:** Jika *noise* sensor kamera (misalnya, ESP32-CAM) cenderung berupa bintik tajam (`salt-and-pepper`), kita mungkin akan beralih ke `cv2.medianBlur` karena lebih efektif untuk jenis *noise* tersebut. Ukuran kernel **(5,5)** adalah titik awal yang baik.