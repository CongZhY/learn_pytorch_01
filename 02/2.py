import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 8.0, 11.0]
# y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w + b

def loss(x_val, y_val):
    y_pred_val = forward(x_val)
    return (y_pred_val - y_val) ** 2

mse_list = []
W = np.arange(0.0, 4.0, 0.1)
B = np.arange(-2.0, 2.1, 0.1)
[w, b] = np.meshgrid(W, B)
l_sum = 0

for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val
    mse_list.append(l_sum / 3)

fig = plt.figure()
ax = Axes3D(fig)
plt.xlabel('w')
plt.ylabel('b')
ax.plot_surface(w, b, l_sum / 3)
plt.show()


