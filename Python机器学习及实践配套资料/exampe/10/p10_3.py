import numpy as np
 
arr = np.asarray([0, 10, 50, 80, 100])
for x in arr:
    x = float(x - arr.mean())/arr.std()
    print (x)
