from operator import is_none

import cv2
import os
import numpy as np

def siapkan_dataset(path_folder, ukuran_wajah=(50, 50)):
    dataset_wajah = []
    label_wajah = []

    path_cascade = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    detektor_wajah = cv2.CascadeClassifier(path_cascade)

    print(f"Membaca dataset dari: {path_folder}")
    for nama_subjek in os.listdir(path_folder):
        path_subjek = os.path.join(path_folder, nama_subjek)

        if not os.path.isdir(path_subjek):
            continue
        

        for nama_file in os.listdir(path_subjek):
            path_gambar = os.path.join(path_subjek, nama_file)
            
            gambar = cv2.imread(path_gambar, cv2.IMREAD_GRAYSCALE)
            if gambar is None:
                continue

            wajah_terdeteksi = detektor_wajah.detectMultiScale(gambar, scaleFactor=1.1, minNeighbors=5)

            if len(wajah_terdeteksi) > 0:
                x, y, w, h = wajah_terdeteksi[0];

                wajah_dicrop = gambar[y:y+h, x:x+w]

                wajah_diresize = cv2.resize(wajah_dicrop, ukuran_wajah)

                wajah_1d = wajah_diresize.flatten()

                dataset_wajah.append(wajah_1d)
                label_wajah.append(nama_subjek)

    matriks_dataset = np.array(dataset_wajah)

    matriks_A_mentah = matriks_dataset.T
    print(f"Selesai! matriks terbentuk dengan ukuran: {matriks_A_mentah.shape}")
    return matriks_A_mentah, label_wajah

if __name__ == "__main__":
    folder_dataset = "test_data"

    if os.path.exists(folder_dataset):
        matriks_A, label = siapkan_dataset(folder_dataset)
        print(f"Total data wajah yang diproses: {len(label)}")
    else:
        print("Folder dataset tidak valid")
