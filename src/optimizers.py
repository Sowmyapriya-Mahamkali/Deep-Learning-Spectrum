import numpy as np

# -------------------
# Gradient Descent
# -------------------
def gradient_descent(W, b, X, y, lr=0.01):
    m = X.shape[0]
    preds = X @ W + b
    error = preds - y
    dW = (1/m) * X.T @ error
    db = (1/m) * np.sum(error)
    W -= lr * dW
    b -= lr * db
    return W, b

# -------------------
# SGD (mini-batch)
# -------------------
def sgd(W, b, X_batch, y_batch, lr=0.01):
    preds = X_batch @ W + b
    error = preds - y_batch
    dW = X_batch.T @ error / X_batch.shape[0]
    db = np.sum(error) / X_batch.shape[0]
    W -= lr * dW
    b -= lr * db
    return W, b

# -------------------
# Momentum, RMSProp, Adam (toy versions)
# -------------------
def momentum_update(W, dW, v, lr=0.01, beta=0.9):
    v = beta*v + (1-beta)*dW
    W -= lr * v
    return W, v

def rmsprop_update(W, dW, s, lr=0.01, beta=0.9, eps=1e-8):
    s = beta*s + (1-beta)*(dW**2)
    W -= lr * dW / (np.sqrt(s)+eps)
    return W, s

def adam_update(W, dW, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1*m + (1-beta1)*dW
    v = beta2*v + (1-beta2)*(dW**2)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    W -= lr * m_hat / (np.sqrt(v_hat)+eps)
    return W, m, v
