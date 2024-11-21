import numpy as np
from scipy.spatial.distance import cdist
from sklearn import datasets



class KMeans():
    def __init__(self, k = 10, metric = "euclidean", max_iter = 300, tol = 1.0e-4):
        """
        Args:
            k (int): 
                The number of centroids you have specified. 
                (number of centroids you think there are)
                
            metric (str): 
                The way you are evaluating the distance between the centroid and data points.
                (euclidean)
                
            max_iter (int): 
                The number of times you want your KMeans algorithm to loop and tune itself.
                
            tol (float): 
                The value you specify to stop iterating because the update to the centroid 
                position is too small (below your "tolerance").
        """ 
        # In the following 4 lines, please initialize your arguments
        self.k = k # The initial clusters we believe it to be.
        self.metric = metric # Store metric to be used.
        self.max_iter = max_iter # Store initial for the amount of iterations.
        self.tol = tol # Store initial parameter for tolerance.

        
        # In the following 2 lines, you will need to initialize 1) centroid, 2) error (set as numpy infinity)
        self.centroid = None # Initializes the centroid to have no value.
        self.error = np.inf # Initialize some immense error here. 
        
    
    def fit(self, matrix: np.ndarray):
        """
        This method takes in a matrix of features and attempts to fit your created KMeans algorithm
        onto the provided data 
        
        Args:
            matrix (np.ndarray): 
                This will be a 2D matrix where rows are your observation and columns are features.
                
                Observation meaning, the specific flower observed.
                Features meaning, the features the flower has: Petal width, length and sepal length width 
        """
        
        # In the line below, you need to randomly select where the centroid's positions will be.
        # Also set your initialized centroid to be your random centroid position. 
        centroid_rand = np.random.choice(matrix.shape[0], size = self.k, replace = False) # Randomly selects centroids from the data points.
        self.centroid = matrix[centroid_rand] # This assigns the random centroids. 


        
        # In the line below, calculate the first distance between your randomly selected centroid positions
        # and the data points
        for _ in range(self.max_iter):
            distances = cdist(self.centroid, matrix, metric=self.metric) # Calculates the distance centroids and the points.
            cluster_assignments = np.argmin(distances, axis=0) # Assigns these distances to the cluster points.

            prev_centroids = self.centroid.copy() # Stores the previous centroids. 

            # This will update the centroids by the use of a for loop to iterate through the number of clusters. 
            for i in range(self.k):
                cluster_points = matrix[cluster_assignments == i]
                if len(cluster_points) > 0:
                    self.centroid[i] = np.mean(cluster_points, axis=0)
       
            # Calculate Inertia or the sum of squared distances between each data point and assigned cluster
            inertia = np.sum([np.sum(np.square(
                matrix[cluster_assignments == i] - self.centroid[i]
            )) for i in range(self.k)])

            # Check for convergence to make sure it approaches a finite value.
            centroid_shift = np.sum(np.sqrt(np.sum((prev_centroids - self.centroid)**2, axis=1)))
            
            # Set error as calculated inertia
            self.error = inertia # Stores the calculated inertia in the self.error variable.

            # Check if change is below tolerance, if not break and exit. 
            if centroid_shift < self.tol: # A simple boolean set-up to test if the shift is within tolerance.
                break
        
            
    
    
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """
        Predicts which cluster each observation belongs to.
        Args:
            matrix (np.ndarray): 
                

        Returns:
            np.ndarray: 
                An array/list of predictions will be returned.
        """
        # In the line below, return data point's assignment 
        return np.argmin(cdist(matrix, self.centroid, metric=self.metric), axis=1) # Returns the data point assignments. 
    
    def get_error(self) -> float:
        """
        The inertia of your KMeans model will be returned

        Returns:
            float: 
                inertia of your fit

        """
        return self.error # Returns the error in terms of the inertia.
    
    
    def get_centroids(self) -> np.ndarray:
    
        """
        Your centroid positions will be returned. 
        """
        # In the line below, return centroid location
        return self.centroid # This will return the centroid location that was converged on.
        

### testing:
""""
iris = datasets.load_iris()
train_data = iris.data
kmeans = KMeans(k = 4)
kmeans.fit(train_data)

for x in range(10):
"""
#data = np.loadtxt("data/iris_extended.csv", delimiter=",")
#print(data)