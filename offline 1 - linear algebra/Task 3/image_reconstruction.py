import cv2
import numpy as np
import matplotlib.pyplot as plt


def low_rank_approximation(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return A_k

# Read the image
image_path = "image.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize for faster computation
resize_dimension = 500
resized_image = cv2.resize(original_image, (resize_dimension, resize_dimension))

# Perform Singular Value Decomposition
U, s, Vt = np.linalg.svd(resized_image, full_matrices=False)

# Vary k from 1 to min(n, m) 
num_values = 10
#k_values = np.linspace(1, min(resize_dimension, resize_dimension), num_values, dtype=int)
k_values= [ 1, 5, 10, 20, 50, 100, 200, 300, 400, 500]
# Plot the resultant k-rank approximations
plt.figure(figsize=(15, 10))
for i, k in enumerate(k_values, 1):
    A_k = low_rank_approximation(resized_image, k)
    
    plt.subplot(2, 5, i)
    plt.imshow(A_k, cmap='gray')
    plt.title(f'k = {k}')
    plt.axis('off')

plt.tight_layout()
plt.show()



