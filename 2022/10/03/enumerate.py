import numpy as np
a = np.array(['Alice','Bob','Carl','Dick','Frank'])
for i,item in enumerate(a,1):
    print('index: {0}, name: {1}'.format(i,item))
