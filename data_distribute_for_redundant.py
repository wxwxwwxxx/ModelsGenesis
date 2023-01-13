import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = np.load("/dataset/generated_cubes/bat_32_64x64x32_0.npy")
# img = np.load("/dataset/ncs_data/x_train_64x64x32.npy")
print(np.shape(img),np.max(img),np.min(img))

for i in range(32):
    plt.subplot(4,8, i+1)
    plt.imshow(img[3231,:,:,i])

plt.show()