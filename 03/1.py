import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [2, 4, 6]

w = 1.0
lr = 0.01

def forward(x):
    return x * w

def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

epoch_list = []
cost_list = []

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= lr * grad_val
    print('epoch: ', epoch, 'w = ', w, 'loss = ', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)

plt.plot(epoch_list, cost_list)
plt.xlabel('epcho')
plt.ylabel('cost')
plt.show()