import numpy as np
from pathlib import Path
import os, sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

data = np.load(f"{RESULTS_DIR}/weights_inputs.npz")

print("x:", data['x'].shape, data['x'].dtype)
print("w_conv mean/std:", data['w_conv'].mean(), data['w_conv'].std())
print("w_lin min/max:", data['w_lin'].min(), data['w_lin'].max())

import matplotlib.pyplot as plt
import numpy as np

data = np.load(f"{RESULTS_DIR}/weights_inputs.npz")
x = data['x']

plt.imshow(x[0,0], cmap="gray")
plt.title("Input image")
plt.colorbar()
plt.savefig("x0.png")
