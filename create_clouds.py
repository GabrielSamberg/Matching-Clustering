
import numpy as np
import matplotlib.pyplot as plt
import pywavefront

scene1 = pywavefront.Wavefront("Data/CAPOD/class1/m1.obj")
scene2 = pywavefront.Wavefront("Data/CAPOD/class1/m7.obj")

X = np.array(scene1.vertices)
Y = np.array(scene2.vertices)

# X = X[1000:]
# Y = Y[1000:]



print(f'This is the shape of X:\n{X.shape} \n\n This is the shape of Y:\n{Y.shape}')