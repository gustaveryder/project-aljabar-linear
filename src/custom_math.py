"""
Fungsi-fungsi kustom:
- Akar kuadrat (Newton-Raphson)
- Norma vektor
- Jarak Euclidean
- Power Iteration + Deflasi untuk eigenvalue/vector terbesar
"""

import numpy as np

def _sqrt_custom(nilai: float) -> float:
    """Akar kuadrat manual menggunakan metode Newton-Raphson."""
    if nilai <= 0.0:
        return 0.0
    x = float(nilai)
    for _ in range(200):
        x_baru = (x + nilai / x) / 2.0
        if abs(x_baru - x) < 1e-12:
            break
        x = x_baru
    return x


def norm_vektor_custom(v: np.ndarray) -> float:
    """Menghitung norma (panjang) vektor secara manual."""
    total = 0.0
    for xi in v:
        total += float(xi) * float(xi)
    return _sqrt_custom(total)


def euclidean_distance_custom(v1: np.ndarray, v2: np.ndarray) -> float:
    """Menghitung jarak Euclidean antara dua vektor secara manual."""
    total = 0.0
    for a, b in zip(v1, v2):
        selisih = float(a) - float(b)
        total += selisih * selisih
    return _sqrt_custom(total)


def _power_iteration(C: np.ndarray, seed: int = 0, max_iter: int = 500) -> tuple:
    """Power Iteration untuk mencari eigenvector dengan eigenvalue terbesar."""
    n = C.shape[0]
    np.random.seed(seed)
    b = np.random.randn(n).astype(np.float64)
    norm0 = norm_vektor_custom(b)
    if norm0 > 1e-12:
        b /= norm0

    for _ in range(max_iter):
        b_baru = C @ b
        n_norm = norm_vektor_custom(b_baru)
        if n_norm < 1e-12:
            break
        b = b_baru / n_norm

    eigenvalue = float(b @ (C @ b))
    return eigenvalue, b


def hitung_eigen_custom(C: np.ndarray, k: int) -> tuple:
    """Menghitung k pasang (eigenvalue, eigenvector) terbesar dari matriks simetri C.
    Metode: Power Iteration + Deflasi Hotelling."""
    M = C.shape[0]
    k = min(k, M)
    eigenvalues = []
    eigenvectors = []
    C_deflasi = C.copy().astype(np.float64)

    for i in range(k):
        eigenval, eigenvec = _power_iteration(C_deflasi, seed=i, max_iter=500)
        eigenvalues.append(eigenval)
        eigenvectors.append(eigenvec)
        outer = np.outer(eigenvec, eigenvec)
        C_deflasi = C_deflasi - eigenval * outer

    return np.array(eigenvalues), np.array(eigenvectors)

