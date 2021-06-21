import numpy as np
import matplotlib.pyplot as plt

x = np.load("C:/Users/sdat7/home/2021/인공지능/[2021 Spring Class] Ajou AI by JBR/test_gallery.npy")
y = np.load("C:/Users/sdat7/home/2021/인공지능/[2021 Spring Class] Ajou AI by JBR/test_query.npy")
z = np.load("C:/Users/sdat7/home/2021/인공지능/[2021 Spring Class] Ajou AI by JBR/train_label.npy")
w = np.load("C:/Users/sdat7/home/2021/인공지능/[2021 Spring Class] Ajou AI by JBR/train.npy")

# print(x.shape)
# print(y.shape)
# print(z.shape)
# print(w.shape)

# print(w[0][0])
plt.imshow(w[0][0])


# print(w[0][0])

# # print(z)
# # print(w[0][0])