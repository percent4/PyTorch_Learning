import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

X = np.linspace(-1, 1, 200)
Y = 0.5 * X + 0.2 * np.random.normal(0, 0.05, (200, ))
X = Variable(torch.Tensor(X.reshape(200,1)))
Y = Variable(torch.Tensor(Y.reshape(200,1)))
print(X)

model = torch.nn.Sequential(
    torch.nn.Linear(1, 1)
    )

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
loss_function = torch.nn.MSELoss()
for i in range(100):
    prediction = model(X)
    loss = loss_function(prediction, Y)
    print("loss:", loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(prediction.data.numpy())
plt.figure(1, figsize=(10, 3))
plt.subplot(131)
plt.title('model')
plt.scatter(X.data.numpy(), Y.data.numpy())
plt.plot(X.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
plt.show()