# 🧠 Pengenalan Wajah dengan Eigenface

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-✓-brightgreen)](https://numpy.org/)
[![Lisensi](https://img.shields.io/badge/Lisensi-MIT-yellow)](LICENSE)

Aplikasi **pengenalan wajah** menggunakan algoritma **Eigenface** (Turk & Pentland, 1991), dibuat dari nol sebagai tugas mata kuliah **Aljabar Linear** di **Universitas Sebelas Maret (UNS)**.

Metode numerik inti — **dekomposisi eigen** dan **jarak Euclidean** — diimplementasikan **secara manual** tanpa menggunakan pustaka aljabar linear siap pakai, sesuai dengan persyaratan tugas.

---

## 📸 Fitur

- **Nilai Eigen & Vektor Eigen Kustom**  
  Power Iteration dengan deflasi Hotelling — tidak menggunakan `numpy.linalg.eig` atau sejenisnya.

- **Jarak Euclidean Kustom**  
  Akar kuadrat manual menggunakan Newton‑Raphson dan loop aritmatika murni.

- **Deteksi Wajah**  
  Menggunakan Haar Cascade bawaan OpenCV untuk mendeteksi dan memotong wajah sebelum pelatihan/pengujian.

- **GUI Modern Bertema Gelap**  
  Dibangun dengan Tkinter; log real‑time, indikator progres, dan pratinjau gambar.

- **Skor Kemiripan**  
  Menampilkan kecocokan terdekat lengkap dengan persentase kemiripan dan jarak Euclidean.

- **Kode Modular**  
  Dipisahkan secara rapi ke dalam modul‑modul independen untuk memudahkan penilaian dan penggunaan ulang.

---

## 📂 Struktur Proyek

```
.
├── custom_math.py       # sqrt manual, norma, jarak Euclidean, Power Iteration + Deflasi
├── dataset.py           # Deteksi wajah, pemotongan, dan pemuatan dataset
├── eigenface.py         # Pelatihan model Eigenface (Turk & Pentland)
├── recognition.py       # Proyeksi dan pencocokan wajah uji
├── gui.py               # GUI Tkinter bertema gelap, redireksi log, threading
├── main.py              # Titik masuk aplikasi
├── requirements.txt     # Dependensi Python
└── README.md            # Dokumentasi ini
```

---

## 🧰 Persyaratan

- Python 3.8 atau lebih baru
- Pustaka berikut (lihat `requirements.txt`):

| Pustaka         | Fungsi                                  |
|-----------------|-----------------------------------------|
| `numpy`         | Operasi matriks dasar                   |
| `opencv-python` | Input/Gambar, deteksi Haar Cascade      |
| `Pillow`        | Menampilkan gambar di GUI               |

- **Tkinter** – biasanya sudah terbundel di Python.  
  Di Linux mungkin perlu diinstal manual:  
  ```bash
  sudo apt-get install python3-tk
  ```

---

## ⚙️ Instalasi

1. **Unduh atau clone repositori**:
   ```bash
   git clone https://github.com/username-anda/eigenface-recognition.git
   cd eigenface-recognition
   ```

2. **(Opsional namun disarankan)** Buat lingkungan virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **Instal dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Cara Menjalankan

```bash
python main.py
```

1. Klik **“Choose Dataset Folder”** dan pilih folder yang berisi gambar‑gambar wajah.  
   Folder tersebut harus memiliki sub‑folder untuk setiap orang (lihat [Struktur Dataset](#-struktur-dataset)).

2. Klik **“Choose Test Image”** dan pilih satu gambar wajah yang ingin dikenali.

3. Klik **“Start Recognition”**. Aplikasi akan:
   - Mendeteksi dan memotong wajah di dalam dataset
   - Melatih model Eigenface (jika belum dilatih)
   - Memproyeksikan gambar uji ke ruang wajah
   - Menampilkan kecocokan terdekat (atau pesan “tidak cocok”)
   - Menampilkan persentase kemiripan, jarak Euclidean, dan waktu pemrosesan.

---

## 🗂 Struktur Dataset

Gunakan dataset wajah dengan susunan folder seperti berikut:

```
dataset_folder/
├── orang_A/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
├── orang_B/
│   ├── img1.jpg
│   └── ...
└── ...
```

Contoh dataset publik yang dapat digunakan untuk pengujian:  
🔗 [Pins Face Recognition di Kaggle](https://www.kaggle.com/datasets/azizbali/football-players-faces-dataset)

> **Catatan**: Program secara otomatis mendeteksi wajah menggunakan OpenCV. Jika tidak ada wajah yang terdeteksi, seluruh gambar akan digunakan.

---

## 🔬 Metodologi

### Pelatihan Eigenface
1. Ubah semua gambar latih menjadi vektor (flatten).
2. Hitung wajah rata‑rata dan kurangkan dari setiap gambar.
3. Bentuk matriks kovarians **L = A ⋅ Aᵀ** (trik Turk & Pentland).
4. Hitung **k** pasangan eigenvalue dan eigenvector terbesar dari **L** menggunakan **Power Iteration kustom dengan deflasi Hotelling**.
5. Dapatkan eigenfaces melalui proyeksi balik.
6. Proyeksikan gambar latih ke ruang eigenface untuk memperoleh vektor bobot (**Ω**).

### Pengenalan
1. Deteksi dan potong wajah dari gambar uji.
2. Kurangkan wajah rata‑rata.
3. Proyeksikan ke ruang wajah untuk memperoleh vektor bobot uji (**Ω_test**).
4. Hitung **jarak Euclidean** (implementasi manual) antara **Ω_test** dan semua **Ω_train**.
5. Gambar latih dengan jarak terkecil di bawah ambang batas dinyatakan sebagai kecocokan.

### Implementasi Kustom
- **`_sqrt_custom`** – Akar kuadrat metode Newton‑Raphson.
- **`euclidean_distance_custom`** – Pengurangan per elemen, pengkuadratan, penjumlahan, lalu akar kuadrat manual.
- **`_power_iteration`** – Metode power klasik untuk pasangan eigen dominan.
- **`hitung_eigen_custom`** – Deflasi berulang untuk mendapatkan banyak pasangan eigen.

Tidak ada `linalg.eig`, `linalg.norm`, atau `math.sqrt` yang digunakan untuk operasi inti tersebut.

---

## 🖥️ Tampilan GUI

*(Tambahkan tangkapan layar GUI ke dalam folder `screenshots/` dan ganti path di bawah jika ada)*

![GUI](screenshots/gui.png)

- **Panel Kiri**: Pemilihan dataset, pemilihan gambar uji, hasil (skor kemiripan, label, jarak), dan tombol kontrol.
- **Panel Tengah**: Tampilan berdampingan gambar uji dan kecocokan terdekat.
- **Panel Kanan**: Log terminal langsung dengan pesan berwarna (dataset, pelatihan, pengenalan, error).

---

## 📝 Kesesuaian dengan Tugas

| Persyaratan | Status |
|-------------|--------|
| 1. Menerima folder dataset & gambar uji | ✅ |
| 2. Sumber dataset bebas (misal Kaggle) | ✅ |
| 3. Menampilkan gambar uji yang dipilih | ✅ |
| 4. Pencocokan dengan Eigenface + jarak Euclidean | ✅ |
| 5. Menampilkan 1 hasil terbaik atau pesan “tidak cocok” | ✅ |
| 6. Perhitungan eigenvalue & eigenvector kustom | ✅ |
| 7. Jarak Euclidean kustom (sqrt manual) | ✅ |

---

## 👥 Kontributor

- **Nama Mahasiswa 1** – `NIM 1`  
- **Nama Mahasiswa 2** – `NIM 2`  
  *Kelas Aljabar Linear D, Universitas Sebelas Maret*


---

## 📄 Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT – lihat file [LICENSE](LICENSE) untuk detail lengkap.

---

## 🙏 Penghargaan

- **M. Turk dan A. Pentland** (1991) – “Eigenfaces for Recognition”
- **OpenCV** untuk pengklasifikasi Haar Cascade.
- Komunitas **Kaggle** untuk dataset wajah publik.
- **Dosen Pengampu** – atas tugas yang menantang dan bimbingannya.
```

---
