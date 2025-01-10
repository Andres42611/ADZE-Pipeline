import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
from collections import Counter

from sklearn.metrics import pairwise_distances_argmin_min, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder

from scipy.optimize import linear_sum_assignment


# Description:
#   Performs K-means clustering on training data, and using calculated centroids predicts
#   labels on testing data. Displays accuracy and shows PCA-reduced testing data along
#   with PCA-reduced centroids
# Accepts:
#   pandas.DataFrame df: Input DataFrame with 'Vector' and 'Class' columns
#   float test_perc: Percentage of data to use for testing (default: 20)
#   int k: Number of clusters for K-means (default: 5)
# Returns:
#   None: Displays plot and prints accuracy

def KMeans_Vis(df, test_perc=20, k=5):
    # Split the data into training and testing sets while preserving class proportions
    train_df, test_df = train_test_split(df, test_size=(test_perc/100), stratify=df['Class'], random_state=42)
    
    # Extract features from the 'Vector' column
    X_train = np.vstack(train_df['Vector'].values)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=1000)

    kmeans.fit(X_train)

    # Get centroids
    centroids = kmeans.cluster_centers_

    # Step 1: Majority Voting
    train_predictions = kmeans.predict(X_train)
    majority_vote_mapping = {}
    for i in range(k):
        cluster_classes = train_df['Class'][train_predictions == i]
        if not cluster_classes.empty:
            majority_vote_mapping[i] = Counter(cluster_classes).most_common(1)[0][0]
        else:
            majority_vote_mapping[i] = None  # Handle empty clusters

    # Step 2: Hungarian Algorithm for global optimization
    true_classes = train_df['Class'].unique()
    cost_matrix = np.zeros((k, len(true_classes)))
    
    # Populate cost matrix
    for i in range(k):
        for j, cls in enumerate(true_classes):
            cost_matrix[i, j] = -np.sum(train_df['Class'][train_predictions == i] == cls)
    
    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create final mapping
    centroid_to_class = {i: true_classes[j] for i, j in zip(row_ind, col_ind)}

    # Predict classes for test data
    X_test = np.stack(test_df['Vector'].values)
    test_predictions = kmeans.predict(X_test)
    predicted_classes = [centroid_to_class[i] for i in test_predictions]

    # Calculate accuracy
    accuracy = accuracy_score(test_df['Class'], predicted_classes)
    print(f"Accuracy: {accuracy:.2f}")

    # Perform PCA on test data and centroids
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_test)
    centroids_pca = pca.transform(centroids)

    # Prepare data for plotting
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Class'] = test_df['Class'].values  # Use .values to ensure proper alignment
    centroids_df = pd.DataFrame(centroids_pca, columns=['PC1', 'PC2', 'PC3'])
    centroids_df['Label'] = [centroid_to_class[i] for i in range(k)]

    # Set up the plot style
    sns.set(style="whitegrid")

    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    pc_pairs = [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC2', 'PC3')]
    # Sort class labels for consistent legend order
    sorted_class_labels = sorted(pca_df['Class'].unique())

    for ax, (pc1, pc2) in zip(axes, pc_pairs):
        # Plot data points
        sns.scatterplot(data=pca_df, x=pc1, y=pc2, hue='Class', hue_order=sorted_class_labels, 
                        palette='deep', alpha=0.35, ax=ax)
        
        # Plot centroids
        sns.scatterplot(data=centroids_df, x=pc1, y=pc2, color='black', marker='X', s=200, ax=ax)
        
        # Annotate centroids with actual class labels
        for _, row in centroids_df.iterrows():
            ax.annotate(row['Label'], (row[pc1], row[pc2]), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontweight='bold',
                        fontsize=12,
                        color='black',
                        )
       
        ax.set_title(f'{pc1} vs {pc2}')
        ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set the main title with total explained variance
    fig.suptitle("3D PCA-Reduced Centroids and Test Data", fontsize=16, y=0.98)
    fig.text(0.5, 0.93, f"Centroid Accuracy on Test Set: {accuracy * 100:.2f}%  |  Total Explained Variance By First 3 PCs: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%", 
             ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()



# Description:
#   Performs label propagation analysis on various distance matrices
#   Uses both unpruned and pruned networks for comparison
# Accepts:
#   pandas.DataFrame df: Input DataFrame with 'Vector' and 'Class' columns
#   int kn: Number of neighbors for kNN in Label Propagation (default: 100)
#   float pruning_percentile: Percentile for pruning distances (default: 90)
# Returns:
#   dict: Dictionary containing accuracy results before and after pruning for each distance metric
def label_prop(df, source, kn=100, pruning_percentile=90):
    # Create a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Encode class labels numerically
    label_encoder = LabelEncoder()
    class_encoded = label_encoder.fit_transform(data['Class'])
    
    # Split data into labeled (20%) and unlabeled (80%) sets
    labeled_data, unlabeled_data = train_test_split(data, test_size=0.8, stratify=data['Class'], random_state=42)
    
    # Prepare labels: encoded for labeled data, -1 for unlabeled data
    labeled_data_labels = label_encoder.transform(labeled_data['Class'])
    unlabeled_data_labels = -1 * np.ones(len(unlabeled_data))

    # Combine labeled and unlabeled data, maintaining order
    all_data = pd.concat([labeled_data, unlabeled_data])
    labels = np.concatenate((labeled_data_labels, unlabeled_data_labels))

    # List of distance matrix files to process
    distance_matrices = [source + '/Results/euclidean_matrix.npy', source + '/Results/std_euclidean_matrix.npy', 
                         source + '/Results/correlation_matrix.npy', source + '/Results/rbf_distances.npy']

    # Initialize results dictionary and create plot
    results = {}
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.ravel()

    # Process each distance matrix
    for idx, matrix_file in enumerate(distance_matrices):
        # Load the distance matrix
        distance_matrix = np.load(matrix_file)
        metric_name = matrix_file.split('_')[0]

        # Set diagonal to large value to avoid self-connections
        max_distance = np.max(distance_matrix)
        np.fill_diagonal(distance_matrix, max_distance*1e6)

        # Perform label propagation on unpruned network
        label_propagation_model = LabelPropagation(kernel='knn', n_neighbors=kn, max_iter=4000)
        label_propagation_model.fit(distance_matrix, labels)

        # Predict and process labels for unlabeled data
        predicted_labels = label_propagation_model.transduction_[-len(unlabeled_data):]
        predicted_labels = np.clip(np.round(predicted_labels).astype(int), 0, len(label_encoder.classes_) - 1)
        predicted_labels = label_encoder.inverse_transform(predicted_labels)
        true_labels = unlabeled_data['Class'].values

        # Calculate accuracy before pruning
        accuracy_before = accuracy_score(true_labels, predicted_labels)

        # Prune the network by setting distances above threshold to a large value to soft-prune
        pruning_threshold = np.percentile(distance_matrix, pruning_percentile)
        pruned_distance_matrix = np.where(distance_matrix > pruning_threshold, distance_matrix, max_distance*1e6)

        # Perform label propagation on pruned network
        label_propagation_model_pruned = LabelPropagation(kernel='knn', n_neighbors=kn, max_iter=3000)
        label_propagation_model_pruned.fit(pruned_distance_matrix, labels)

        # Predict and process labels for pruned network
        predicted_labels_pruned = label_propagation_model_pruned.transduction_[-len(unlabeled_data):]
        predicted_labels_pruned = np.clip(np.round(predicted_labels_pruned).astype(int), 0, len(label_encoder.classes_) - 1)
        predicted_labels_pruned = label_encoder.inverse_transform(predicted_labels_pruned)

        # Calculate accuracy after pruning
        accuracy_after = accuracy_score(true_labels, predicted_labels_pruned)

        # Store results
        results[metric_name] = {'before': accuracy_before, 'after': accuracy_after}

        # Plot pruned weight distribution
        sns.histplot(pruned_distance_matrix[pruned_distance_matrix <= max_distance].flatten(), 
                     kde=False, ax=axs[idx])
        axs[idx].set_title(f'{metric_name.capitalize()} (Accuracy: {accuracy_after:.2f})')
        axs[idx].set_xlabel('Distance')
        axs[idx].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    return results