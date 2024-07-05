import numpy as np

def generate_invertible_symmetric_matrix(n):
    while True:
        A = np.random.randint(-9,  9, size=(n, n))
        A = (A + A.T) // 2
        determinant = np.linalg.det(A)
        if determinant != 0:
            break   
    return A


n = int(input("Enter the dimension of the matrix (n): "))
random_invertible_symmetric_matrix = generate_invertible_symmetric_matrix(n)

print("Random Invertible Symmetric Matrix A:" )
print(random_invertible_symmetric_matrix)

# Perform eigen decomposition
eigen_values, eigen_vectors = np.linalg.eig(random_invertible_symmetric_matrix)
print("Eigenvalues:", eigen_values)
print("\nEigenvectors:", eigen_vectors)



# Reconstruct A from eigenvalues and eigenvectors
random_invertible_symmetric_matrix_reconstructed = eigen_vectors @ np.diag(eigen_values) @ np.linalg.inv(eigen_vectors)

# Check if the reconstruction worked properly
is_reconstruction_correct = np.allclose(random_invertible_symmetric_matrix, random_invertible_symmetric_matrix_reconstructed)
if is_reconstruction_correct:
    print("Reconstruction is correct.")
else:
    print("Reconstruction is not correct.")
    
