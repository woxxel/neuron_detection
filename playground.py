
import matplotlib.pyplot as plt

## casting component n to dense matrix and displaying it
A = onacid.estimates.A[:,n].reshape(512,512).todense()
plt.figure(); plt.imshow(A); plt.show()
