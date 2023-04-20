from scipy import misc

img = misc.face()

import numpy as np
from numpy import linalg

import time


img_array = img / 255
img_gray = img_array @ [0.2126, 0.7152, 0.0722]

def run_all():
    U, s, Vt = linalg.svd(img_gray)
    Sigma = np.zeros((U.shape[1], Vt.shape[0]))
    np.fill_diagonal(Sigma, s)
    linalg.norm(img_gray - U @ Sigma @ Vt)
    img_array_transposed = np.transpose(img_array, (2, 0, 1))

if __name__ == '__main__':
    print("==================")
    print("Experiment: nd-array.py\n")

    time_list = []
    for _ in range(50):
        t0 = time.time()
        U, s, Vt = linalg.svd(img_gray)
        time_list.append(time.time()-t0)
    print(f"SVD takes {sum(time_list)/len(time_list)}s on average")

    Sigma = np.zeros((U.shape[1], Vt.shape[0]))
    np.fill_diagonal(Sigma, s)

    time_list = []
    for _ in range(50):
        t0 = time.time()
        linalg.norm(img_gray - U @ Sigma @ Vt)
        time_list.append(time.time()-t0)
    print(f"Norm takes {sum(time_list)/len(time_list)}s on average")

    time_list = []
    for _ in range(50):
        t0 = time.time()
        img_array_transposed = np.transpose(img_array, (2, 0, 1))
        time_list.append(time.time()-t0)
    print(f"Transpose takes {sum(time_list)/len(time_list)}s on average")
    print("==================\n")
