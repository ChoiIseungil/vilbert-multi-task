import numpy as np 
import os 
from PIL import Image

# 아 넘파이가 저장된 피처 정보지.... 이게 이미지는 아님! 착각함

NPYPATH = "../../mnt/nas2/seungil/features/GA/9_Sweet potato cultivation in Polynesia.npy"




if __name__ == "__main__" : 
    A = np.load(NPYPATH, allow_pickle=True)
    print("shape : ", A.shape)
    # im = Image.fromarray(A)
    # im.save("./9_Sweet potato cultivation in Polynesia.jpg")