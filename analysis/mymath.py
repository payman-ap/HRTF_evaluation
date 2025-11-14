# mymath.py
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt



# try out new SNR method:

def sum(a,b):
    return a + b


if __name__=="__main__":
    print(f'summation {sum(7,8):.2f}')


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



a = np.array([1,2,3])
b = np.array([4,5,6,7])

# h = sp.signal.convolve(a,b)
h = np.convolve(a,b)


M = len(a)
N = len(b)
L = M + N - 1


result = np.zeros(L)

n = 0
while n < L:
    k = 0
    while k < M :
        
        # check index
        if (k <= n) and ((n - k) < N):
            result[n] += a[k] * b[n - k]
            print(f"+ n: {n}, k: {k}, n-k: {n - k}")
        else:
            print(f"- n: {n}, k: {k}, n-k: {n - k}")
        
        k += 1
        
    n += 1


print(f"result = {result}")



print(f"a = {a}")
print(f"b = {b}")
print(f"h = {h}")






# a = np.linspace(0,2*np.pi,1000)
# b = np.sin(2*np.pi*a)

# plt.plot(a,b)
# plt.show()







