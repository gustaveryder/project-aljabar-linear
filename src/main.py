import cv2
import os
import numpy as np

def siapkan_dataset(path_folder, ukuran_wajah=(50, 50)):
    dataset_wajah = []
    label_wajah = []

