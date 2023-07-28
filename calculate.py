import numpy as np
import cv2
import math
import time
from config import IMAGE_SIZE, ROTATION_RESOLUTION

IMAGE_DIAGONAL = math.sqrt(2) * IMAGE_SIZE


def calculate_intensity(x, y, px, py, cx, cy):
    # line x = px + cx * t
    # line y = py + cy * t

    # Calculate distance from point to line
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    d = abs((cx * (py - y) - cy * (px - x)) / math.sqrt(cx * cx + cy * cy))

    # Calculate intensity
    i = max(0, 1 - d)

    return i


def rasterize_line(x1, x2, y1, y2):
    # Calculate dx and dy
    dx = x2 - x1
    dy = y2 - y1

    # Initialize starting points
    x = x1
    y = y1

    # Rasterize line
    points = []  # array of [x, y, intensity]

    if abs(dx) > abs(dy):
        # Line is more horizontal than vertical
        if dx < 0:
            # Swap points
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            dx = -dx
            dy = -dy

        for x in range(math.floor(x1), math.ceil(x2 + 1)):
            y = y1 + dy * (x - x1) / dx
            y = math.floor(y)
            intensity = calculate_intensity(x, y, x1, y1, dx, dy)
            points.append([x, y, intensity])
    else:
        # Line is more vertical than horizontal
        if dy < 0:
            # Swap points
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            dx = -dx
            dy = -dy
        for y in range(math.floor(y1), math.ceil(y2 + 1)):
            x = x1 + dx * (y - y1) / dy
            x = math.floor(x)
            intensity = calculate_intensity(x, y, x1, y1, dx, dy)
            points.append([x, y, intensity])

    return points


# Read image
image = cv2.imread("./test.png")

# Make image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize it to IMAGE_SIZE x IMAGE_SIZE
image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

# Save it to file
cv2.imwrite("./test_resized.png", image)

# Convert image to float32 nupy array
image = np.float32(image) / 255.0

# Sample matrix
samples = []
# Detection vector
detections = []

# Simulate X-ray
start_time = time.time()
for t1 in range(ROTATION_RESOLUTION):
    theta = math.pi * t1 / ROTATION_RESOLUTION  # from 0 to pi
    print("theta = " + str(theta))

    l = math.ceil(IMAGE_DIAGONAL / 2 + 1)
    lx = math.cos(theta) * l
    ly = math.sin(theta) * l

    RAY_RESOLUTION = math.ceil(IMAGE_DIAGONAL + 1)
    for t2 in range(RAY_RESOLUTION):
        d = (
            t2 / RAY_RESOLUTION - 0.5
        ) * IMAGE_DIAGONAL  # from -IMAGE_DIAGONAL / 2 to IMAGE_DIAGONAL / 2
        dx = math.sin(theta) * d
        dy = -math.cos(theta) * d

        # Calculate ray start and end points
        sx = lx + dx + IMAGE_SIZE / 2
        sy = ly + dy + IMAGE_SIZE / 2
        ex = dx - lx + IMAGE_SIZE / 2
        ey = dy - ly + IMAGE_SIZE / 2

        points = rasterize_line(sx, ex, sy, ey)

        # Sample vector
        sample = np.zeros(IMAGE_SIZE * IMAGE_SIZE)
        detection = 0
        for [x, y, intensity] in points:
            if x < 0 or x >= IMAGE_SIZE or y < 0 or y >= IMAGE_SIZE:
                continue
            sample[x + y * IMAGE_SIZE] = intensity
            detection += image[y, x]

        samples.append(sample)
        detections.append(detection)
end_time = time.time()
print("Simulated X-ray in " + str(end_time - start_time) + " seconds.")

# Convert to numpy array
samples = np.array(samples)
detections = np.array(detections)

# Print size of samples, detections
print(samples.shape)
print(detections.shape)

# Save samples and detections to file
np.save("./test_samples.npy", samples)
np.save("./test_detections.npy", detections)

# Save samples as a large image file
image = samples * 255
image = np.uint8(image)
cv2.imwrite("./test_samples.png", image)

"""
let samples := S (samples * cells), detections := D (samples * 1)
for unknown vector x (cells * 1), we want to solve the matrix equation S * x = D.
"""

# Solve the matrix equation with pseudo-inverse
print("Solving matrix equation...")
start_time = time.time()
x = np.linalg.lstsq(samples, detections, rcond=None)[0]
end_time = time.time()
print("Solved in " + str(end_time - start_time) + " seconds.")

# Print size of x
print(x.shape)

# Save x to file
np.save("./test_result.npy", x)

# Run visualize.py to visualize the result
