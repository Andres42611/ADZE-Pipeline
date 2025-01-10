import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns



# Description:
#   Calculates multiple distance metrics between vectors and saves the results
#   Computes Euclidean, standardized Euclidean, correlation, and RBF kernel distances
# Accepts:
#   numpy.ndarray vectors: Input vectors for distance calculation
#   str source: Path to save the output files
#   float gamma: Parameter for RBF kernel (default: 0.0006)
# Returns:
#   dict: Dictionary containing distance metrics (condensed and matrix forms)
def calculate_multiple_distances(df, source, gamma=0.0006):
    
    vectors = np.stack(df['Vector'].values)
    
    # Compute Euclidean distances
    # Uses scipy's pdist function to calculate pairwise distances
    euclidean_distances = pdist(vectors, metric='euclidean')
    # Convert condensed distance matrix to square form
    euclidean_matrix = squareform(euclidean_distances)
    
    # Compute standardized Euclidean distances
    # Calculate variances for each feature
    variances = np.var(vectors, axis=0, ddof=1)
    # Compute standardized Euclidean distances using feature variances
    std_euclidean_distances = pdist(vectors, metric='seuclidean', V=variances)
    # Convert to square form
    std_euclidean_matrix = squareform(std_euclidean_distances)
    
    # Compute correlation distances
    # Uses Pearson correlation coefficient to calculate distances
    correlation_distances = pdist(vectors, metric='correlation')
    # Convert to square form
    correlation_matrix = squareform(correlation_distances)
    
    # Compute RBF kernel distances
    # Calculate RBF kernel matrix
    K = rbf_kernel(vectors, gamma=float(gamma))
    # Extract diagonal elements of kernel matrix
    K_diag = np.diag(K)
    # Compute RBF distances using kernel trick
    rbf_distances = np.sqrt(K_diag[:, np.newaxis] + K_diag[np.newaxis, :] - 2 * K)
    # Extract upper triangular part of the distance matrix
    rbf_distances_condensed = rbf_distances[np.triu_indices(len(vectors), k=1)]
    
    # Save distance matrices to .npy files
    # Store results for later use or analysis
    np.save(source + '/Results/euclidean_matrix.npy', euclidean_matrix)
    np.save(source + '/Results/std_euclidean_matrix.npy', std_euclidean_matrix)
    np.save(source + '/Results/correlation_matrix.npy', correlation_matrix)
    np.save(source + '/Results/rbf_distances.npy', rbf_distances)
    
    # Return dictionary containing all computed distances
    # Each entry contains both condensed and matrix forms of the distances
    return {
        'euclidean': (euclidean_distances, euclidean_matrix),
        'standardized_euclidean': (std_euclidean_distances, std_euclidean_matrix),
        'correlation': (correlation_distances, correlation_matrix),
        'rbf_kernel': (rbf_distances_condensed, rbf_distances)
    }



# Description:
#   Plots histograms of various distance metrics distributions
#   Creates a 2x2 grid of subplots, each showing a different distance metric distribution
# Accepts:
#   dict distance_dict: Dictionary containing distance metrics (output from calculate_multiple_distances)
#   str source: Path to save the output plot
# Returns:
#   None: Displays the plot and saves it as a PNG file
def plot_distance_distributions(distance_dict, source):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), tight_layout=True)
    
    # Set the main title for the entire figure
    fig.suptitle('Distributions of Distance Metrics', fontsize=16)

    # Define distance metrics and their display titles
    metrics_titles = [
        ('euclidean', 'Euclidean'),
        ('standardized_euclidean', 'Standardized Euclidean'),
        ('correlation', 'Correlation'),
        ('rbf_kernel', 'RBF Kernel')
    ]

    # Iterate over each metric
    for i, (metric, title) in enumerate(metrics_titles):
        # Calculate the row and column for the current subplot
        row, col = divmod(i, 2)
        ax = axs[row, col]
        # Get he condensed distance array for the current metric
        distances = distance_dict[metric][0]

        # Plot histogram of distances using seaborn
        sns.histplot(distances, kde=False, ax=ax, color='skyblue', edgecolor='black')
        
        # Set title and labels for the subplot
        ax.set_title(f'{title} Distances')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        
        # Add a light grid to the subplot
        ax.grid(alpha=0.2)

    # Save the figure as a PNG file
    plt.savefig(source+'/Results/distance_distribs.png')
    # Display the plot
    plt.show()