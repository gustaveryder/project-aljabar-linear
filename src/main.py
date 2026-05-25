"""
==========================================================
APLIKASI PENGENALAN WAJAH - EIGENFACE
Mata Kuliah : Aljabar Linear Kelas D
Universitas Sebelas Maret
==========================================================
Catatan penting:
  - Fungsi eigen (eigenvalue & eigenvector) → DITULIS SENDIRI (Power Iteration + Deflasi)
  - Fungsi euclidean distance              → DITULIS SENDIRI (Newton-Raphson sqrt)
  - Operasi matriks biasa (perkalian dll)  → boleh pakai numpy
  - Deteksi & resize wajah                → OpenCV (diizinkan)
==========================================================
"""

import tkinter as tk
from gui import FaceRecognitionApp

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()