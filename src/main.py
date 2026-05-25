"""
APLIKASI PENGENALAN WAJAH - EIGENFACE
Mata Kuliah : Aljabar Linear Kelas D
Universitas Sebelas Maret
"""

import tkinter as tk
from gui import FaceRecognitionApp

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()