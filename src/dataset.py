"""
Membaca & mempersiapkan dataset wajah:
- Deteksi wajah dengan Haar Cascade (bawaan OpenCV)
- Resize ke ukuran seragam
- Mengembalikan matriks data, label, dan path asli
"""

import os
import cv2
import numpy as np


def _crop_wajah(gambar_gray: np.ndarray, detektor) -> np.ndarray:
    """Coba deteksi wajah dengan beberapa konfigurasi toleransi."""
    configs = [
        dict(scaleFactor=1.1,  minNeighbors=5, minSize=(20, 20)),
        dict(scaleFactor=1.1,  minNeighbors=3, minSize=(15, 15)),
        dict(scaleFactor=1.05, minNeighbors=1, minSize=(10, 10)),
    ]
    for cfg in configs:
        hasil = detektor.detectMultiScale(gambar_gray, **cfg)
        if len(hasil) > 0:
            x, y, w, h = hasil[0]
            return gambar_gray[y:y + h, x:x + w]
    return gambar_gray


def _kumpulkan_gambar(path_folder: str) -> list:
    """Kumpulkan semua path gambar beserta labelnya."""
    EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
    entri = []
    isi_folder = sorted(os.listdir(path_folder))
    ada_subfolder = any(
        os.path.isdir(os.path.join(path_folder, e)) for e in isi_folder
    )

    if ada_subfolder:
        for nama_subjek in isi_folder:
            path_subjek = os.path.join(path_folder, nama_subjek)
            if not os.path.isdir(path_subjek):
                continue
            for nama_file in sorted(os.listdir(path_subjek)):
                if os.path.splitext(nama_file)[1].lower() not in EXT:
                    continue
                entri.append((os.path.join(path_subjek, nama_file), nama_subjek))
    else:
        label = os.path.basename(path_folder)
        for nama_file in isi_folder:
            if os.path.splitext(nama_file)[1].lower() not in EXT:
                continue
            entri.append((os.path.join(path_folder, nama_file), label))

    return entri


def siapkan_dataset(path_folder: str, ukuran_wajah: tuple = (50, 50)) -> tuple:
    """Membaca semua gambar dari folder dataset, mendeteksi wajah."""
    dataset_wajah = []
    label_wajah   = []
    path_gambar   = []

    path_cascade = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    detektor_wajah = cv2.CascadeClassifier(path_cascade)

    print(f"[Dataset] Membaca dari: {path_folder}")

    daftar_gambar = _kumpulkan_gambar(path_folder)
    if not daftar_gambar:
        raise ValueError(
            f"Tidak ditemukan file gambar di folder:\n{path_folder}\n\n"
            "Pastikan folder berisi subfolder per orang yang masing-masing\n"
            "berisi file gambar (.jpg/.png/dll)."
        )

    gagal_baca = 0
    for path_img, label in daftar_gambar:
        gambar = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
        if gambar is None:
            gagal_baca += 1
            continue

        roi           = _crop_wajah(gambar, detektor_wajah)
        wajah_resize  = cv2.resize(roi, ukuran_wajah)
        wajah_1d      = wajah_resize.flatten().astype(np.float64)

        dataset_wajah.append(wajah_1d)
        label_wajah.append(label)
        path_gambar.append(path_img)

    if not dataset_wajah:
        raise ValueError(
            "Tidak ada gambar yang berhasil dibaca dari dataset!\n\n"
            f"Total file ditemukan : {len(daftar_gambar)}\n"
            f"Gagal dibaca          : {gagal_baca}\n\n"
            "Periksa apakah file gambar tidak rusak dan format didukung."
        )

    matriks_dataset = np.array(dataset_wajah)
    print(
        f"[Dataset] Selesai. Diproses: {len(label_wajah)} gambar "
        f"| Gagal baca: {gagal_baca} "
        f"| Ukuran matriks: {matriks_dataset.shape}"
    )
    return matriks_dataset, label_wajah, path_gambar