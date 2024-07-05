#load dataset using numpy
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def plot_data_PCA(dataset):
    
    
    # Extract the x and y coordinates
    x_values = dataset[:, 0]
    y_values = dataset[:, 1]  
    
    # Create a scatter plot
    plt.scatter(x_values, y_values, marker='.', color='blue', alpha=0.5)
    plt.title('After PCA')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.savefig('PCA.png')
    plt.show()

def plot_data(dataset):
    
    
    # Extract the x and y coordinates
    x_values = dataset[:, 0]
    y_values = dataset[:, 1]  
    
    # Create a scatter plot
    plt.scatter(x_values, y_values, marker='.', color='blue', alpha=0.5)
    plt.title('Original Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.savefig('PCA.png')
    plt.show()

def plot_data_EM(dataset, cluster_labels):
    
    
    # Extract the x and y coordinates
    x_values = dataset[:, 0]
    y_values = dataset[:, 1]

    # Create a scatter plot colorful cluster
    plt.scatter(x_values, y_values, marker='.', c=cluster_labels ,cmap='rainbow', alpha=0.5)
    
    #name each color as cluster
    plt.legend(handles=plt.scatter(x_values, y_values, marker='.', c=cluster_labels ,cmap='rainbow', alpha=0.5).legend_elements()[0], labels=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7', 'Cluster 8'])

    
    plt.title('After applying EM')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.savefig('EM.png')
    plt.show()



def PCA(data):
    #load dataset from txt file


    # Now, 'data' contains the loaded dataset as a NumPy array
    #print(data)

    ##############################task 1 ######################
    #compute mean in each and every ROW
    mean = np.mean(data, axis=0)
    #print("mean is: ", mean)
    #subtract mean from each and every ROW
    data = data - mean

    #SVD decomposition
    U, S, V = np.linalg.svd(data, full_matrices=True)


    k = 3
    data= np.dot(data, (V[:k, :].T))
   
    return data


################################task 2#####################

def log_likelihood_func(data, mu, sigma, pi):
    log_likelihood = 0
    for i in range(data.shape[0]):
        temp = 0
        for j in range(len(pi)):
            # Add regularization to ensure positive definiteness of covariance matrix
            sigma[j] = (sigma[j] + sigma[j].T) / 2.0  # Ensure symmetry
            sigma[j] = sigma[j] + np.identity(sigma[j].shape[0]) * 1e-6  # Add a small regularization term

            temp += pi[j] * multivariate_normal.pdf(data[i], mu[j], sigma[j])
        log_likelihood += np.log(temp)
    return log_likelihood


def EM_algorithm(data, k, max_iter, regularization=1e-6):
    # Initialize mu, sigma, pi with random positive values
    mu = np.random.rand(k, data.shape[1])
    sigma = np.array([np.eye(data.shape[1])] * k)
    pi = np.random.rand(k)
    pi = pi / pi.sum()

    for i in range(max_iter):
        # E-step
        gamma = np.zeros((data.shape[0], k))
        for j in range(k):
            gamma[:, j] = pi[j] * multivariate_normal.pdf(data, mu[j], sigma[j])
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        # M-step
        Nk = np.sum(gamma, axis=0)
        for j in range(k):
            mu[j] = 1 / Nk[j] * np.sum(gamma[:, j] * data.T, axis=1).T
            sigma[j] = 1 / Nk[j] * np.dot((gamma[:, j] * (data - mu[j]).T), (data - mu[j]))

            # Add regularization to ensure positive semidefiniteness
            sigma[j] = (sigma[j] + sigma[j].T) / 2.0  # Ensure symmetry
            sigma[j] = sigma[j] + regularization * np.identity(data.shape[1])

            pi[j] = Nk[j] / data.shape[0]

    return mu, sigma, pi, gamma, data
        
        

if __name__ == '__main__':
    
    #take k as a list from 3 to 8
    k = [3,4,5,6,7,8]
    file_path = '100D_data_points.txt'
    # Load the dataset using np.loadtxt with the correct delimiter
    data = np.loadtxt(file_path, delimiter=',')
    #count number of rows and columns
    print("shape of data is: ", data.shape)
    if data.shape[1] > 2:
        data = PCA(data)
        plot_data_PCA(data)
        
    else:
        plot_data(data)
   
    max_likelihood = -np.inf
    max_likelihood_ever = -np.inf
    
    #for each k value run the em algo for 5 times
    for i in k:
        for j in range(5):
            mu, sigma, pi, gamma, data = EM_algorithm(data, i, 100)
            log_likelihood = log_likelihood_func(data, mu, sigma, pi)
            if max_likelihood < log_likelihood:
                max_likelihood = log_likelihood
        print( " k: ", i, " max log_likelihood: ", log_likelihood)
        #find best k
        if max_likelihood_ever < max_likelihood:
            max_likelihood_ever = max_likelihood
            best_k = i
            
    #print best k
    print("best k is: ", best_k)
    #plot 
    mu, sigma, pi,gamma, fdata = EM_algorithm(data, best_k, 100)
    cluster_labels = np.argmax(gamma, axis=1)
    plot_data_EM(fdata, cluster_labels)    
        
    