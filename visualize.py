import numpy as np
import cv2
from config import IMAGE_SIZE

# Load x from file
x = np.load("./test_result.npy")

# Reshape x to IMAGE_SIZE x IMAGE_SIZE
x = x.reshape(IMAGE_SIZE, IMAGE_SIZE)

# Print min, max of x
print("min:", np.min(x))
print("max:", np.max(x))

# Clip x to [0, 1]
x = np.clip(x, 0, 1)

# Save x as an image file
image = x * 255
image = np.uint8(image)
cv2.imwrite("./test_result.png", image)
