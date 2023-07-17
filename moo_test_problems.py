import numpy as np

def MOP2(x):
    n = len(x)
    z1 = 1 - np.exp(-np.sum((x - 1/np.sqrt(n))**2))
    z2 = 1 - np.exp(-np.sum((x + 1/np.sqrt(n))**2))
    return np.array([z1, z2])

def MOP4(x):
    a = 0.8
    b = 3
    z1 = np.sum(-10 * np.exp(-0.2 * np.sqrt(x[:-1]**2 + x[1:]**2)))
    z2 = np.sum(np.abs(x)**a + 5 * np.sin(x)**b)
    return np.array([z1, z2])
