import numpy as np
import pandas as pd

a = [1,2,3]
aa = np.array(a)
print(aa, aa.shape)

b = [[1,2], [10, 20], [50, 60]]
bb = np.array(b)
print('B', bb.shape)
print('B count', bb.shape[0])
print('B dim', bb.shape[1])
print(bb.reshape((2,3)))
print( bb.reshape((-1)) )
print(bb.reshape((-1,1)))

df = pd.read_csv('apple.csv', header=0)
print(df.values.shape[0])



