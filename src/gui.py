"""
GUI modern bertema gelap menggunakan Tkinter.
Menangani redirect stdout, tampilan dataset, gambar uji, hasil, dan log.
"""

import cv2
import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import time
import threading
import queue
import io

# Impor dari modul lain
from dataset import siapkan_dataset
from eigenface import training_eigenface
from recognition import kenali_wajah

# ── Palet Warna ─────────────────────────────────────────────
C = {
    'bg':        '#0d1117',   # hitam biru sangat gelap
    'panel':     '#161b22',   # panel sedikit lebih terang
    'card':      '#1c2128',   # kartu/item
    'border':    '#30363d',   # border halus
    'accent':    '#58a6ff',   # biru aksen (GitHub-style)
    'accent2':   '#3fb950',   # hijau aksen (sukses)
    'danger':    '#f85149',   # merah (error)
    'warning':   '#d29922',   # kuning (warning)
    'text':      '#e6edf3',   # teks utama
    'text_dim':  '#8b949e',   # teks redup
    'text_code': '#79c0ff',   # teks log/code
    'btn':       '#21262d',   # tombol normal
    'btn_hover': '#30363d',   # tombol hover
    'green_btn': '#238636',   # tombol hijau
    'green_h':   '#2ea043',   # tombol hijau hover
    'log_bg':    '#0d1117',   # background log terminal
}

UKURAN_WAJAH = (50, 50)
K_KOMPONEN   = 50

# Font
F_TITLE  = ("Consolas", 20, "bold")
F_SUB    = ("Consolas", 11, "bold")
F_BODY   = ("Consolas", 9)
F_LOG    = ("Consolas", 9)
F_BADGE  = ("Consolas", 14, "bold")


class LogRedirector(io.TextIOBase):
    """Redirect stdout ke queue agar bisa ditampilkan di GUI."""
    def __init__(self, log_queue: queue.Queue):
        self._queue = log_queue

    def write(self, text: str):
        if text.strip():
            self._queue.put(text.rstrip())
        return len(text)

    def flush(self):
        pass


class FaceRecognitionApp:
    """Aplikasi GUI pengenalan wajah berbasis Eigenface — Dark Modern UI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Eigenface · Face Recognition — Kelompok Aljabar Linear")
        self.root.geometry("1200x750")
        self.root.configure(bg=C['bg'])
        self.root.resizable(True, True)
        self.root.minsize(900, 600)

        # State
        self.folder_dataset   = None
        self.path_gambar_uji  = None
        self.mean_wajah       = None
        self.eigenfaces       = None
        self.omega_training   = None
        self.label_wajah      = None
        self.path_paths       = None
        self.is_trained       = False

        # Log queue
        self._log_queue = queue.Queue()
        self._redirector = LogRedirector(self._log_queue)
        sys.stdout = self._redirector

        self._build_gui()
        self._poll_log()

    # ----------------------------------------------------------
    # LOG POLLING — baca queue & tulis ke terminal widget
    # ----------------------------------------------------------

    def _poll_log(self):
        """Cek log queue setiap 100ms, tulis ke terminal widget."""
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _append_log(self, text: str):
        """Tambahkan baris ke terminal log widget."""
        self.txt_log.configure(state='normal')

        # Warna berdasarkan prefix
        if text.startswith("[Dataset]"):
            tag = "dataset"
        elif text.startswith("[Training]"):
            tag = "training"
        elif text.startswith("[Rekognisi]"):
            tag = "rekognisi"
        elif text.startswith("[Error]") or "Error" in text or "error" in text:
            tag = "error"
        else:
            tag = "normal"

        import time as _t
        ts = _t.strftime("%H:%M:%S")
        self.txt_log.insert('end', f"[{ts}] ", "timestamp")
        self.txt_log.insert('end', text + "\n", tag)
        self.txt_log.see('end')
        self.txt_log.configure(state='disabled')

    # ----------------------------------------------------------
    # BUILD GUI
    # ----------------------------------------------------------

    def _build_gui(self):
        # ── Header ─────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=C['panel'], height=56)
        hdr.pack(fill='x', side='top')
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text="EIGENFACE RECOGNITION",
            font=("Consolas", 16, "bold"),
            bg=C['panel'], fg=C['accent']
        ).pack(side='left', padx=20, pady=14)

        tk.Label(
            hdr, text="Aljabar Linear D · Universitas Sebelas Maret",
            font=("Consolas", 9),
            bg=C['panel'], fg=C['text_dim']
        ).pack(side='right', padx=20)

        # Garis tipis di bawah header
        tk.Frame(self.root, bg=C['border'], height=1).pack(fill='x')

        # ── Body ────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=C['bg'])
        body.pack(fill='both', expand=True, padx=16, pady=12)

        # Kolom kiri (kontrol)
        self._build_sidebar(body)
        # Kolom tengah (gambar)
        self._build_image_panel(body)
        # Kolom kanan (log terminal)
        self._build_log_panel(body)

    # ----------------------------------------------------------

    def _build_sidebar(self, parent):
        """Panel kiri: dataset, test image, hasil, tombol."""
        sidebar = tk.Frame(parent, bg=C['panel'], width=240,
                           relief='flat', bd=0)
        sidebar.pack(side='left', fill='y', padx=(0, 10))
        sidebar.pack_propagate(False)

        pad = {'padx': 14}

        # ── Section: Dataset ─────────────────────────────────
        self._section_label(sidebar, "DATASET")

        self.btn_dataset = self._make_button(
            sidebar, "Choose Dataset Folder",
            self._pilih_dataset, style='accent'
        )
        self.btn_dataset.pack(fill='x', **pad, pady=(0, 4))

        self.lbl_dataset = tk.Label(
            sidebar, text="No folder selected",
            bg=C['panel'], fg=C['text_dim'],
            font=F_BODY, wraplength=210, justify='left'
        )
        self.lbl_dataset.pack(anchor='w', **pad, pady=(0, 8))

        # Badge status training
        self.badge_trained = tk.Label(
            sidebar, text="  NOT TRAINED  ",
            bg=C['warning'], fg='#000',
            font=("Consolas", 8, "bold"), padx=6, pady=2
        )
        self.badge_trained.pack(anchor='w', **pad, pady=(0, 12))

        tk.Frame(sidebar, bg=C['border'], height=1).pack(fill='x', padx=14, pady=4)

        # ── Section: Test Image ──────────────────────────────
        self._section_label(sidebar, "TEST IMAGE")

        self.btn_testimg = self._make_button(
            sidebar, "Choose Test Image",
            self._pilih_gambar_uji, style='normal'
        )
        self.btn_testimg.pack(fill='x', **pad, pady=(0, 4))

        self.lbl_gambar = tk.Label(
            sidebar, text="No file selected",
            bg=C['panel'], fg=C['text_dim'],
            font=F_BODY, wraplength=210, justify='left'
        )
        self.lbl_gambar.pack(anchor='w', **pad, pady=(0, 12))

        tk.Frame(sidebar, bg=C['border'], height=1).pack(fill='x', padx=14, pady=4)

        # ── Section: Hasil ───────────────────────────────────
        self._section_label(sidebar, "RESULT")

        self.lbl_pct_val = tk.Label(
            sidebar, text="—",
            bg=C['panel'], fg=C['text'],
            font=("Consolas", 28, "bold")
        )
        self.lbl_pct_val.pack(anchor='w', **pad, pady=(2, 0))

        tk.Label(sidebar, text="similarity score",
                 bg=C['panel'], fg=C['text_dim'],
                 font=("Consolas", 8)).pack(anchor='w', **pad, pady=(0, 4))

        self.lbl_match = tk.Label(
            sidebar, text="",
            bg=C['panel'], fg=C['text_dim'],
            font=F_BODY, wraplength=210, justify='left'
        )
        self.lbl_match.pack(anchor='w', **pad, pady=(0, 12))

        tk.Frame(sidebar, bg=C['border'], height=1).pack(fill='x', padx=14, pady=4)

        # ── Tombol Start ─────────────────────────────────────
        self.btn_start = self._make_button(
            sidebar, "Start Recognition",
            self._mulai_pengenalan, style='green'
        )
        self.btn_start.pack(fill='x', padx=14, pady=(10, 4))

        # ── Status + Waktu ───────────────────────────────────
        self.lbl_status = tk.Label(
            sidebar, text="Ready",
            bg=C['panel'], fg=C['text_dim'],
            font=("Consolas", 8), wraplength=210
        )
        self.lbl_status.pack(anchor='w', padx=14, pady=2)

        self.lbl_waktu = tk.Label(
            sidebar, text="",
            bg=C['panel'], fg=C['accent'],
            font=("Consolas", 8)
        )
        self.lbl_waktu.pack(anchor='w', padx=14, pady=2)

        # Progress bar
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Dark.Horizontal.TProgressbar",
            troughcolor=C['bg'],
            background=C['accent'],
            thickness=3
        )
        self.progress = ttk.Progressbar(
            sidebar, mode='indeterminate', length=210,
            style="Dark.Horizontal.TProgressbar"
        )
        self.progress.pack(padx=14, pady=6, fill='x')

    # ----------------------------------------------------------

    def _build_image_panel(self, parent):
        """Panel tengah: gambar uji dan hasil terbaik."""
        mid = tk.Frame(parent, bg=C['bg'])
        mid.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Labels atas
        row_lbl = tk.Frame(mid, bg=C['bg'])
        row_lbl.pack(fill='x', pady=(0, 6))

        tk.Label(row_lbl, text="TEST IMAGE",
                 font=("Consolas", 10, "bold"),
                 bg=C['bg'], fg=C['text_dim']).pack(side='left', expand=True)
        tk.Label(row_lbl, text="CLOSEST MATCH",
                 font=("Consolas", 10, "bold"),
                 bg=C['bg'], fg=C['text_dim']).pack(side='right', expand=True)

        # Frame gambar
        row_img = tk.Frame(mid, bg=C['bg'])
        row_img.pack(fill='both', expand=True)

        self.frame_img_uji = tk.Frame(
            row_img, bg=C['card'],
            relief='flat', bd=0,
            highlightthickness=1, highlightbackground=C['border']
        )
        self.frame_img_uji.pack(side='left', fill='both', expand=True, padx=(0, 6))

        self.lbl_img_uji = tk.Label(
            self.frame_img_uji, bg=C['card'],
            text="No image\nselected", fg=C['text_dim'],
            font=("Consolas", 10)
        )
        self.lbl_img_uji.pack(expand=True, fill='both')

        self.frame_img_hasil = tk.Frame(
            row_img, bg=C['card'],
            relief='flat', bd=0,
            highlightthickness=1, highlightbackground=C['border']
        )
        self.frame_img_hasil.pack(side='right', fill='both', expand=True, padx=(6, 0))

        self.lbl_img_hasil = tk.Label(
            self.frame_img_hasil, bg=C['card'],
            text="Awaiting\nrecognition", fg=C['text_dim'],
            font=("Consolas", 10)
        )
        self.lbl_img_hasil.pack(expand=True, fill='both')

    # ----------------------------------------------------------

    def _build_log_panel(self, parent):
        """Panel kanan: terminal log output."""
        log_frame = tk.Frame(parent, bg=C['panel'], width=340,
                             highlightthickness=1, highlightbackground=C['border'])
        log_frame.pack(side='right', fill='y')
        log_frame.pack_propagate(False)

        # Header terminal
        term_hdr = tk.Frame(log_frame, bg='#1a1a2e', height=32)
        term_hdr.pack(fill='x')
        term_hdr.pack_propagate(False)

        # Dot dekorasi ala macOS
        dot_frame = tk.Frame(term_hdr, bg='#1a1a2e')
        dot_frame.pack(side='left', padx=10, pady=8)
        for col in ['#ff5f56', '#ffbd2e', '#27c93f']:
            tk.Label(dot_frame, text="●", fg=col, bg='#1a1a2e',
                     font=("Consolas", 8)).pack(side='left', padx=2)

        tk.Label(
            term_hdr, text="console · stdout",
            font=("Consolas", 8), bg='#1a1a2e', fg=C['text_dim']
        ).pack(side='left', padx=6)

        # Tombol clear di kanan
        tk.Button(
            term_hdr, text="clear",
            command=self._clear_log,
            bg='#1a1a2e', fg=C['text_dim'],
            font=("Consolas", 7), bd=0, cursor='hand2',
            activebackground='#2a2a3e', activeforeground=C['text'],
            relief='flat'
        ).pack(side='right', padx=10)

        # Text widget log
        self.txt_log = tk.Text(
            log_frame,
            bg=C['log_bg'], fg=C['text_code'],
            font=F_LOG,
            bd=0, relief='flat',
            wrap='word',
            state='disabled',
            selectbackground=C['border'],
            insertbackground=C['accent'],
            padx=10, pady=8,
            spacing1=2, spacing3=2
        )
        self.txt_log.pack(fill='both', expand=True)

        # Scrollbar
        sb = ttk.Scrollbar(log_frame, orient='vertical',
                           command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)

        # Tags warna per kategori
        self.txt_log.tag_config("timestamp", foreground=C['text_dim'])
        self.txt_log.tag_config("dataset",   foreground='#79c0ff')
        self.txt_log.tag_config("training",  foreground='#d2a8ff')
        self.txt_log.tag_config("rekognisi", foreground='#56d364')
        self.txt_log.tag_config("error",     foreground=C['danger'])
        self.txt_log.tag_config("normal",    foreground=C['text'])

        # Pesan welcome
        self._append_log("System initialized. Ready.")
        self._append_log("Select a dataset folder and test image,")
        self._append_log("then click 'Start Recognition'.")

    # ----------------------------------------------------------
    # HELPERS UI
    # ----------------------------------------------------------

    def _section_label(self, parent, text: str):
        tk.Label(
            parent, text=text,
            bg=C['panel'], fg=C['text_dim'],
            font=("Consolas", 8, "bold")
        ).pack(anchor='w', padx=14, pady=(12, 4))

    def _make_button(self, parent, text: str, cmd, style='normal') -> tk.Button:
        cfg = {
            'normal': dict(bg=C['btn'],       fg=C['text'],    activebackground=C['btn_hover']),
            'accent': dict(bg=C['accent'],    fg='#0d1117',    activebackground='#79aeff'),
            'green':  dict(bg=C['green_btn'], fg='white',      activebackground=C['green_h']),
        }[style]

        return tk.Button(
            parent, text=text,
            command=cmd,
            font=("Consolas", 9, "bold"),
            relief='flat', bd=0, padx=10, pady=7,
            cursor='hand2',
            activeforeground=cfg.get('fg', 'white'),
            **{k: v for k, v in cfg.items()}
        )

    def _clear_log(self):
        self.txt_log.configure(state='normal')
        self.txt_log.delete('1.0', 'end')
        self.txt_log.configure(state='disabled')

    # ----------------------------------------------------------
    # AKSI TOMBOL
    # ----------------------------------------------------------

    def _pilih_dataset(self):
        folder = filedialog.askdirectory(title="Pilih Folder Dataset Wajah")
        if folder:
            self.folder_dataset = folder
            self.is_trained = False
            name = os.path.basename(folder)
            self.lbl_dataset.config(text=f"{name}", fg=C['text'])
            self.badge_trained.config(text="  ○  NOT TRAINED  ", bg=C['warning'])
            self._set_status("Dataset dipilih. Siap diproses.")

    def _pilih_gambar_uji(self):
        path = filedialog.askopenfilename(
            title="Pilih Gambar Uji",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                       ("All files", "*.*")]
        )
        if path:
            self.path_gambar_uji = path
            self.lbl_gambar.config(text=f"{os.path.basename(path)}", fg=C['text'])
            self._tampilkan_gambar(path, self.lbl_img_uji, self.frame_img_uji)

    def _mulai_pengenalan(self):
        if not self.folder_dataset:
            messagebox.showwarning("Peringatan", "Pilih folder dataset terlebih dahulu!")
            return
        if not self.path_gambar_uji:
            messagebox.showwarning("Peringatan", "Pilih gambar uji terlebih dahulu!")
            return

        self.btn_start.config(state='disabled')
        self.btn_dataset.config(state='disabled')
        self.btn_testimg.config(state='disabled')
        self.progress.start(10)
        self._set_status("Memproses…")
        self.lbl_pct_val.config(text="…", fg=C['text_dim'])

        t = threading.Thread(target=self._proses_background, daemon=True)
        t.start()

    # ----------------------------------------------------------
    # PROSES BACKGROUND
    # ----------------------------------------------------------

    def _proses_background(self):
        try:
            waktu_mulai = time.time()

            if not self.is_trained:
                self._after(self._set_status, "Menyiapkan dataset…")
                matriks, self.label_wajah, self.path_paths = siapkan_dataset(
                    self.folder_dataset, UKURAN_WAJAH
                )
                self._after(self._set_status, "Training Eigenface (harap tunggu)…")
                k = min(K_KOMPONEN, len(self.label_wajah) - 1)
                self.mean_wajah, self.eigenfaces, self.omega_training, _ = \
                    training_eigenface(matriks, k_komponen=k)
                self.is_trained = True
                self._after(
                    lambda: self.badge_trained.config(
                        text="  TRAINED  ", bg=C['accent2'], fg='#000'
                    )
                )

            self._after(self._set_status, "Mengenali wajah…")
            label, path_cocok, jarak, pct = kenali_wajah(
                self.path_gambar_uji,
                self.mean_wajah, self.eigenfaces, self.omega_training,
                self.label_wajah, self.path_paths
            )

            durasi = time.time() - waktu_mulai
            self._after(self._update_hasil, label, path_cocok, jarak, pct, durasi)

        except Exception as e:
            print(f"[Error] {e}")
            self._after(messagebox.showerror, "Error", str(e))
        finally:
            self._after(self._selesai_proses)

    # ----------------------------------------------------------
    # UPDATE GUI
    # ----------------------------------------------------------

    def _update_hasil(self, label, path_cocok, jarak, pct, durasi):
        self.lbl_waktu.config(text=f"{durasi:.3f}s")

        if label is None:
            self.lbl_pct_val.config(text="0%", fg=C['danger'])
            self.lbl_match.config(
                text=f"Tidak ada wajah cocok\nJarak: {jarak:.2f}",
                fg=C['danger']
            )
            self.lbl_img_hasil.config(image='', text="No match\nfound", fg=C['danger'])
            self.lbl_img_hasil.image = None
        else:
            color = C['accent2'] if pct >= 50 else C['warning']
            self.lbl_pct_val.config(text=f"{pct:.1f}%", fg=color)
            self.lbl_match.config(
                text=f"Label : {label}\nEuclidean dist : {jarak:.2f}",
                fg=C['text']
            )
            self._tampilkan_gambar(path_cocok, self.lbl_img_hasil, self.frame_img_hasil)

        self._set_status("Selesai")

    def _selesai_proses(self):
        self.progress.stop()
        self.btn_start.config(state='normal')
        self.btn_dataset.config(state='normal')
        self.btn_testimg.config(state='normal')

    # ----------------------------------------------------------
    # UTILITAS
    # ----------------------------------------------------------

    def _after(self, fn, *args):
        self.root.after(0, lambda: fn(*args))

    def _set_status(self, teks: str):
        self.lbl_status.config(text=teks)

    def _tampilkan_gambar(self, path: str, lbl: tk.Label, frame: tk.Frame):
        try:
            img = Image.open(path)
            frame.update_idletasks()
            fw = frame.winfo_width()  or 320
            fh = frame.winfo_height() or 380
            img.thumbnail((fw - 10, fh - 10), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            lbl.config(image=photo, text='')
            lbl.image = photo
        except Exception as e:
            lbl.config(text=f"Gagal muat:\n{e}", image='', fg=C['danger'])