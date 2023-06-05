

import numpy as np
import matplotlib.pyplot as plt


data = np.load('scalars.npz')

plt.figure(figsize=(8,6))
plt.subplot(211)
plt.plot(data['t'], data['E'])
plt.grid()
plt.xlabel('t')
plt.ylabel('Energy')
plt.subplot(212)
plt.plot(data['t'], data['H'])
plt.grid()
plt.xlabel('t')
plt.ylabel('Helicity')
plt.tight_layout()
plt.savefig('scalars.pdf')


