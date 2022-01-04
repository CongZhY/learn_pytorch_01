import torch

x_data = [1, 2, 3]
y_data = [2, 4, 6]

w = torch.tensor([1.0])
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) ** 2

lr = 0.01

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad', x, y, w.grad.item())
        w.data -= lr * w.grad.data
        w.grad.data.zero_()
    print('progress:', epoch, l.item())
print('predict ', 4, forward(4).item)