import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from kmeans import KMeans

def plot_3d_clusters(data, predictions, model, score):
    # 3D scatter plot of the data points colored by their cluster
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=predictions, cmap='viridis')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title(f'KMeans Clustering\nSilhouette Score: {score:.2f}')
    
    # Add a legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    
    plt.show()

def generate_elbow_plot(data, max_k=10):
    inertias = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(k=k)
        kmeans.fit(data)
        inertias.append(kmeans.get_error())
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

def evaluate_kmeans(data, k_range=range(2, 11)):
    scores = {}
    
    for k in k_range:
        kmeans = KMeans(k=k)
        kmeans.fit(data)
        predictions = kmeans.predict(data)
        
        # Ensure at least 2 unique clusters for silhouette score
        if len(np.unique(predictions)) > 1:
            score = silhouette_score(data, predictions)
            scores[k] = score
    
    return scores

def main():
    # Load example dataset
    iris = pd.read_csv("data/iris_extended.csv")
    
    # Select numeric features
    numeric_features = iris.select_dtypes(include=[np.number]).columns.tolist()
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(iris[numeric_features])
    
    # Use first 3 features for 3D visualization
    X_3d = X[:, :3]

    # Determine optimal number of clusters
    scores = evaluate_kmeans(X)
    best_k = max(scores, key=scores.get)
    print(f"Optimal number of clusters: {best_k}")
    print("Silhouette Scores:", scores)
    
    # Generate Elbow Plot
    generate_elbow_plot(X)
    
    # Perform clustering with optimal k
    kmeans = KMeans(k=best_k)
    kmeans.fit(X_3d)
    predictions = kmeans.predict(X_3d)
    
    # Calculate silhouette score for best k
    silhouette_avg = silhouette_score(X_3d, predictions)
    
    # 3D Visualization
    plot_3d_clusters(X_3d, predictions, kmeans, silhouette_avg)

if __name__ == "__main__":
    main()