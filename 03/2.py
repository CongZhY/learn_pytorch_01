import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [2, 4, 6]

w = 1.0
lr = 0.01

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y)

epoch_list = []
loss_list = []
lr = 0.01

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= grad * lr
        print('\tgrad: ', x, y, grad)
        l = loss(x, y)
    print('epoch: ', epoch, 'w = ', w, 'loss = ', l)
    epoch_list.append(epoch)
    loss_list.append(l)

plt.plot(epoch_list, loss_list)
plt.xlabel('epcho')
plt.ylabel('loss')
plt.show()