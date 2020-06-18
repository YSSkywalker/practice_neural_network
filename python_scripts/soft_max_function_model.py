import numpy as np
import matplotlib.pylab as plt

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # anti-overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

a = np.array([3.2, 2.2, 4.5])
y = softmax(a)

print(y)
