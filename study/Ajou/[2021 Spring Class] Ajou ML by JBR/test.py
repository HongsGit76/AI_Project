import numpy as np
import torch

x = np.load("C:/Users/sdat7/home/2021/기계학습/[2021 Spring Class] Ajou ML by JBR/train_data.npy", allow_pickle=True)
y = np.load("C:/Users/sdat7/home/2021/기계학습/[2021 Spring Class] Ajou ML by JBR/train_label.npy", allow_pickle=True)

# print(len(x))
print(y[13])

t = 0
ft = 0
with open("C:/Users/sdat7/home/git_commit/AI_Project/study/Ajou/[2021 Spring Class] Ajou ML by JBR/label.txt","r") as f:
    while True:
        line = f.readline()
        if not line: break
        answers = line.split(' ')
        for i in answers:
            a = i.split(",")
            if len(a) == 1: break
            if y[int(a[0])] == a[1]: t+=1
            else:
                print(f"{y[int(a[0])]}, {a[1]}")
                ft+=1
percent = t/(t+ft)

print(percent)
