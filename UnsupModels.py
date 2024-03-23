import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations

# Description:
#     Scales numerical data within a pandas DataFrame using the StandardScaler from
#     scikit-learn, which standardizes features by removing the mean and scaling to unit variance.
# Accepts:
#     pandas.DataFrame df: The DataFrame containing the data to be scaled. It is assumed that this
#                          DataFrame consists of numerical features; however, the user can specify
#                          columns to exclude from scaling.
#     list columns_to_exclude=None: An optional list of column names to be excluded from scaling.
#                                   If no columns are to be excluded, this parameter can be omitted
#                                   or set to an empty list.
# Returns:
#     numpy.ndarray X_scaled: A NumPy array containing the scaled values of the DataFrame's numerical
#                             features, with the mean removed and scaled to unit variance. The order
#                             of the features in `X_scaled` corresponds to the order of columns in
#                             `columns_to_keep`, which are the DataFrame columns not listed in
#                             `columns_to_exclude`.
def scale_data(df, columns_to_exclude=None):
    if columns_to_exclude is None:
        columns_to_exclude = []
    scaler = StandardScaler()
    columns_to_keep = df.columns.difference(columns_to_exclude)
    X_scaled = scaler.fit_transform(df[columns_to_keep])
    return X_scaled

# Description:
#     This function performs Principal Component Analysis (PCA) on scaled data to reduce its dimensionality
#     to a specified number of principal components. It can optionally plot the results for visual analysis.
#     The function supports plotting the first two PCA components against each other for up to 3 dimensions
#     directly and any two-component combination for higher dimensions. It also reports the variance explained
#     by each principal component.
# Accepts:
#     numpy.ndarray X_scaled: The scaled input data to be transformed using PCA. This data should already
#                             be preprocessed (e.g., scaled or normalized) to ensure effective dimensionality reduction.
#     int n_components=2: The number of principal components to retain. This value must be at least 2.
#                         It determines the new dimensionality of the data after PCA transformation.
#     str title='PCA-Reduced Data': The base title for any generated plots. This title is augmented
#                                    with specific component comparisons when plotting.
#     bool plot_results=True: Flag to determine whether the PCA-transformed data should be plotted.
#                             If True, plots are generated and displayed. If False, no plots are shown,
#                             and the function returns the PCA-transformed data, the PCA model, and
#                             the variance ratios.
# Returns:
#     If plot_results is False, the function returns a tuple containing:
#         - numpy.ndarray X_pca: The PCA-transformed data, with dimensionality reduced to 'n_components'.
#         - PCA pca: The PCA model object that was fitted to the input data.
#         - numpy.ndarray variance_ratios: An array containing the variance explained by each of the
#                                           retained principal components.
#     If plot_results is True, the function does not return a value but instead generates and displays
#     plots for visual analysis of the PCA-transformed data.
def PCA_plot(X_scaled, n_components=2, title='PCA-Reduced Data', plot_results = True):
    if n_components < 2:
        raise ValueError("Number of components must be at least 2.")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    variance_ratios = pca.explained_variance_ratio_

    if plot_results:
      # When the number of components is 2 or 3, handle as special cases
      if n_components <= 3:
        pairs = combinations(range(n_components), 2)
      else:
        # For more than 3 components, generate all unique 2-tuple combinations
        pairs = combinations(range(n_components), 2)
      for pair in pairs:
          plt.figure(figsize=(8, 6))
          plt.scatter(X_pca[:, pair[0]], X_pca[:, pair[1]], alpha=0.7)
          plt.xlabel(f'PCA Component {pair[0]+1}')
          plt.ylabel(f'PCA Component {pair[1]+1}')
          plt.title(f'{title}: Component {pair[0]+1} vs. Component {pair[1]+1}')
          plt.grid(alpha=0.2)
          plt.show()
      # Print variance explained by each principal component
      for i, ratio in enumerate(variance_ratios, 1):
          print(f"Variance explained by the {i}th principal component:", ratio)
          print(f"Total variance explained by the first {n_components} principal components:", sum(variance_ratios[:n_components]))
    
    else:
      return X_pca, pca, variance_ratios

# Description:
#     This function calculates the coordinates of feature vectors in 2-dimensional space for each pair of
#     principal components generated by a PCA transformation.
# Accepts:
#     PCA pca: The PCA object from scikit-learn after it has been fitted to data.
#     list feature_names: A list of strings representing the names of the features in the original data.
#     int n_components: The number of principal components that were retained during the PCA transformation.
# Returns:
#     tuple: A tuple containing two elements:
#         - list component_pairs: A list of tuples, each representing a pair of principal components
#                                 for which the feature vectors were calculated.
#         - dict vector_coordinates: A dictionary where each key is a tuple representing a pair of
#                                    principal components, and each value is a list of dictionaries.
#                                    Each nested dictionary represents a feature vector, containing
#                                    'feature_name', 'x', and 'y' keys. 'feature_name' corresponds to
#                                    the name of the feature, while 'x' and 'y' represent the feature's
#                                    coordinates on the principal components specified by the key.
def calculate_feature_vectors(pca, feature_names, n_components):
    vector_coordinates = {}
    # Convert component_pairs to a list immediately to support len() and multiple iterations
    component_pairs = list(combinations(range(n_components), 2)) if n_components > 2 else [(0, 1)]

    for pair in component_pairs:
        comp_x, comp_y = pair
        vectors_2d = [{'feature_name': feature_names[i], 'x': pca.components_[comp_x][i], 'y': pca.components_[comp_y][i]} for i in range(len(feature_names))]
        vector_coordinates[pair] = vectors_2d

    return component_pairs, vector_coordinates

# Description:
#     This function performs PCA on subsets of data (replicates) from a larger dataset
#     and plots the PCA-transformed data points for selected replicates alongside feature vectors representing
#     the original features in the PCA-reduced space. The function allows highlighting specific replicates
#     by plotting them in distinct colors, while the rest are plotted in a uniform, less prominent color.
# Accepts:
#     pandas.DataFrame df: The complete dataset from which subsets (replicates) will be selected for PCA
#                          analysis.
#     list selected_replicates: A list of integers representing the replicate numbers to be highlighted
#                               in the PCA plots.
#     int n_components=2: The number of principal components to retain in the PCA analysis. Defaults to 2,
#                         which is suitable for 2D plotting (but can be any integer > 2)
#     str title_prefix='PCA Overlay: ': A prefix for the plot titles.
# Returns:
#     None: This function does not return a value. Its primary purpose is to generate and display plots
#           for visual analysis.
def plot_replicates_pca(df, selected_replicates, n_components=2, title_prefix='PCA Overlay: '):
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_replicates)))
    
    # Scale the full dataset, excluding 'Class' if present
    X_scaled_full = scale_data(df, ['Class'] if 'Class' in df.columns else [])
    
    # Perform PCA on the full dataset without plotting results, ensure PCA_plot returns the PCA object too
    _, pca, _ = PCA_plot(X_scaled_full, n_components=n_components, plot_results=False)
    
    # Ensure feature_names excludes 'Class' if present
    feature_names = df.drop(columns=['Class'], errors='ignore').columns.tolist()
    
    # Calculate feature vectors for the full dataset
    pairs, vector_coordinates = calculate_feature_vectors(pca, feature_names, n_components)
    
    for pair in pairs:
        # Create a new figure for each pair
        plt.figure(figsize=(12, 10))
        ax = plt.gca()  # Get current axis
        
        for i in range(0, len(df), 198):  # Iterate over each replicate
            temp_df = df.iloc[i:i+198]
            X_scaled = scale_data(temp_df, ['Class'] if 'Class' in temp_df.columns else [])
            X_pca, _, _ = PCA_plot(X_scaled, n_components=n_components, plot_results=False)
            highlight = (i // 198 + 1) in selected_replicates
            color = colors[selected_replicates.index(i // 198 + 1)] if highlight else 'darkgray'
            alpha = 0.8 if highlight else 0.2

            # Plot data points for each replicate
            ax.scatter(X_pca[:, pair[0]], X_pca[:, pair[1]], color=color, alpha=alpha, zorder=(2 if highlight else 1), label=f'Replicate #{i // 198 + 1}' if highlight else None)
        
        # Overlay feature vectors on the plot
        for vector in vector_coordinates[pair]:
            ax.arrow(0, 0, vector['x']*3, vector['y']*3, color='black', width=0.002, head_width=0.05, length_includes_head=True, zorder=3)
            ax.text(vector['x']*(3+0.75), vector['y']*(3+0.75), vector['feature_name'], color='black', fontsize=9, zorder=3)
        
        ax.set_xlabel(f'PC{pair[0]+1}')
        ax.set_ylabel(f'PC{pair[1]+1}')
        ax.set_title(f'{title_prefix}PC{pair[0]+1} vs. PC{pair[1]+1}')
        ax.grid(alpha=0.2)
        ax.legend(loc='best')
        
        plt.show()
