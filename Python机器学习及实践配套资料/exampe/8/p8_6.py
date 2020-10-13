import numpy as np
import scipy.sparse as sparse

r=np.array([0,3,1,2,6,3,6,3,4])
c=np.array([0,0,2,2,2,4,5,6,3])
data=np.array([1,1,1,1,1,1,1,1,1])
a = np.ones(7)
sparse_matrix =sparse.coo_matrix((data, (r,c)), shape=(7,7))

print(sparse_matrix)
print(sparse_matrix.todense())
M = sparse_matrix.dot(a)
print(M)
