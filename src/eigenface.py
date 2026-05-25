"""
Pelatihan model Eigenface berdasarkan Turk & Pentland (1991).
Menggunakan fungsi kustom dari custom_math.
"""

import numpy as np
from custom_math import hitung_eigen_custom, norm_vektor_custom


def training_eigenface(matriks_dataset: np.ndarray, k_komponen: int = 50) -> tuple:
    """Melatih model Eigenface."""
    M, N = matriks_dataset.shape
    print(f"[Training] Jumlah gambar: {M}, dimensi tiap gambar: {N}")

    mean_wajah = np.mean(matriks_dataset, axis=0)
    A = matriks_dataset - mean_wajah
    L = A @ A.T

    k_actual = min(k_komponen, M - 1)
    print(f"[Training] Menghitung {k_actual} komponen eigen (Power Iteration + Deflasi)...")
    eigenvalues, eigenvectors_L = hitung_eigen_custom(L, k_actual)

    eigenfaces = []
    for i in range(k_actual):
        v_i = eigenvectors_L[i]
        u_i = A.T @ v_i
        norm_u = norm_vektor_custom(u_i)
        if norm_u > 1e-12:
            u_i /= norm_u
        eigenfaces.append(u_i)

    eigenfaces = np.array(eigenfaces)
    omega_training = A @ eigenfaces.T

    print(f"[Training] Selesai. Eigenfaces shape: {eigenfaces.shape}")
    return mean_wajah, eigenfaces, omega_training, eigenvalues