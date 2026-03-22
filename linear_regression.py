import numpy as np
import matplotlib.pyplot as plt

# dataset (house size vs price)
X = np.array([500, 800, 1000, 1200, 1500, 1800, 2000])
y = np.array([100, 160, 200, 240, 300, 360, 400])

# normalize
X = X / 1000

# initialize parameters
w = 0
b = 0
alpha = 0.1
epochs = 100

m = len(X)

# store cost for graph
costs = []

# training
for _ in range(epochs):
    y_hat = w * X + b
    
    # cost (for visualization)
    cost = (1/(2*m)) * np.sum((y_hat - y)**2)
    costs.append(cost)
    
    # gradients
    dw = (1/m) * np.sum((y_hat - y) * X)
    db = (1/m) * np.sum(y_hat - y)
    
    # update
    w = w - alpha * dw
    b = b - alpha * db

# final values
print("Final w:", w)
print("Final b:", b)

# prediction line
x_line = np.linspace(min(X), max(X), 100)
y_line = w * x_line + b

# plot data + line
plt.scatter(X, y)
plt.plot(x_line, y_line)
plt.xlabel("House Size (normalized)")
plt.ylabel("Price")
plt.title("Linear Regression Fit")
plt.show()

# cost graph
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Time")
plt.show()