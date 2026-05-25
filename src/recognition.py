"""
Pengenalan wajah dengan proyeksi ke ruang eigen.
Mengembalikan label terdekat, path gambar cocok, jarak, dan persentase kemiripan.
"""

import os
import cv2
import numpy as np
from custom_math import euclidean_distance_custom


def kenali_wajah(path_gambar_uji: str,
                 mean_wajah: np.ndarray,
                 eigenfaces: np.ndarray,
                 omega_training: np.ndarray,
                 label_wajah: list,
                 path_paths: list,
                 ukuran_wajah: tuple = (50, 50),
                 threshold: float = 8000.0) -> tuple:
    """Mengenali wajah dari gambar uji."""
    path_cascade = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    detektor_wajah = cv2.CascadeClassifier(path_cascade)

    gambar = cv2.imread(path_gambar_uji, cv2.IMREAD_GRAYSCALE)
    if gambar is None:
        raise ValueError(f"Gagal membaca gambar: {path_gambar_uji}")

    wajah_terdeteksi = detektor_wajah.detectMultiScale(
        gambar, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )

    if len(wajah_terdeteksi) > 0:
        x, y, w, h = wajah_terdeteksi[0]
        roi = gambar[y:y + h, x:x + w]
        print(f"[Rekognisi] Wajah terdeteksi di ({x},{y}) ukuran {w}x{h}")
    else:
        roi = gambar
        print("[Rekognisi] Wajah tidak terdeteksi, menggunakan seluruh gambar")

    wajah_diresize = cv2.resize(roi, ukuran_wajah)
    wajah_1d = wajah_diresize.flatten().astype(np.float64)
    wajah_norm = wajah_1d - mean_wajah
    omega_uji = eigenfaces @ wajah_norm

    jarak_minimum = float('inf')
    index_terdekat = -1

    for i, omega_i in enumerate(omega_training):
        jarak = euclidean_distance_custom(omega_uji, omega_i)
        if jarak < jarak_minimum:
            jarak_minimum = jarak
            index_terdekat = i

    print(f"[Rekognisi] Jarak minimum: {jarak_minimum:.2f} | Threshold: {threshold:.2f}")

    if jarak_minimum > threshold:
        print("[Rekognisi] Hasil: TIDAK ADA WAJAH YANG COCOK (melebihi threshold)")
        return None, None, jarak_minimum, 0.0

    percentage = max(0.0, (1.0 - jarak_minimum / threshold)) * 100.0
    label_terdekat = label_wajah[index_terdekat]
    print(f"[Rekognisi] Hasil: {label_terdekat} | Kemiripan: {percentage:.2f}%")

    return label_terdekat, path_paths[index_terdekat], jarak_minimum, percentage