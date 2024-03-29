import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Description:
#    This function processes a given DataFrame by splitting string representations of vectors into individual components, 
#    converting them into separate float columns, and optionally filtering the dataset to only include 'mu' (mean) related columns.
#    It assumes vectors are stored as comma-separated strings in the DataFrame's columns. Each vector component is separated 
#    into new columns with the suffix '_1'(for means), '_2' (for variance), and '_3'(for SE)., and the original string columns are removed from the DataFrame. If 
#    'only_mu' is set to True, it retains only the first component (the mean 'mu') of each vector, drops 
#    the other components (var and SE).
# Accepts:
#    DataFrame df: The dataset to process, which may contain vector data stored as comma-separated string values in its columns.
#    bool only_mu: A flag indicating whether to retain only the first component of each vector. Default is True.
# Returns:
#    None: The function modifies the DataFrame in place, meaning the passed 'df' is altered directly and there is no return value.
def process_dataset(df, only_mu=True):
    #seperate mu, var, and SE vectors
    for column in df.columns:
        if df[column].dtype == object and column != 'Class':
            expanded_columns = df[column].str.split(',', expand=True)
            expanded_columns = expanded_columns.astype(float)  # Convert to float
            expanded_column_names = [f"{column}_{i+1}" for i in range(expanded_columns.shape[1])]
            df[expanded_column_names] = expanded_columns
            df.drop(column, axis=1, inplace=True)  # Drop original column
    
    #process dataset
    if mu:
        columns_to_drop = [col for col in df.columns if col[-1] not in ['1', 'k', 's', 'e']]
    else:
        columns_to_drop = []
        
    df_final = df.drop(columns=columns_to_drop)
    df = df_final.copy()
    

# Description:
#    (3/6/24 Weekly Doc PCA plot function)
#    This function performs Principal Component Analysis (PCA) on scaled feature data and plots the first two PCA components, 
#    color-coded by class labels and annotate select k values. It can handle any number of dimensions. It calculates the explained variance ratio 
#    for the components and optionally prints it. The function can also return the
#    transformed PCA data, the PCA object, and the variance ratios if plotting is not requested.
# Accepts:
#    DataFrame df: the input dataset containing features along with 'Class' and 'Replicate' columns.
#    array-like y: Corresponding class labels for the data points in X_scaled.
#    array-like k_values: Corresponding k labels for the data points in X_scaled.
#    int n_components: The number of principal components to compute and plot. Default is 2.
#    str title: The title to display on the plot. Default is 'PCA-Reduced Data'.
#    bool plot_results: A flag to determine if the function should plot the results or return the PCA data. Default is True.
# Returns:
#    When plot_results is True, the function does not return anything; it shows the PCA plot directly.
#    When plot_results is False, it returns a tuple containing:
#        numpy.ndarray X_pca: The transformed PCA data.
#        PCA pca: The fitted PCA object from scikit-learn.
#        numpy.ndarray variance_ratios: The explained variance ratio for each computed component.
def PCA_plot(df, y, k_values, n_components=2, title='PCA-Reduced Data', 
             special_k_values=[2, 50, 100, 150, 199], plot_results=True):
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(['Class', 'Replicate'], axis=1))
    if n_components < 2:
        raise ValueError("Number of components must be at least 2.")
    if len(X_scaled) != len(k_values):
        raise ValueError("Length of `k_values` must match the number of samples in `X_scaled`.")

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    variance_ratios = pca.explained_variance_ratio_

    if plot_results:
        plt.figure(figsize=(8, 6))
        unique_classes = np.unique(y)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
        class_color_dict = dict(zip(unique_classes, colors))

        # Plot all points
        for class_label, color in class_color_dict.items():
            mask = y == class_label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], color=color, alpha=0.4, label=class_label)

        # Annotate special k values
        for k in special_k_values:
            for class_label in unique_classes:
                mask = (y == class_label) & (k_values == k)
                if any(mask):
                    index = np.where(mask)[0][0]  # Use the first matching index
                    plt.scatter(X_pca[index, 0], X_pca[index, 1], color='black', s=50, alpha=0.7)
                    plt.text(X_pca[index, 0], X_pca[index, 1], f'k={k}', color='black', fontsize=9, alpha=0.9)

        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(title)
        plt.grid(alpha=0.2)
        plt.legend()
        plt.show()

        # Print variance explained by each principal component
        for i, ratio in enumerate(variance_ratios, 1):
            print(f"Variance explained by the {i}th principal component:", ratio)
        print(f"Total variance explained by the first {n_components} components:", np.sum(variance_ratios[:n_components]))
    else:
        return X_pca, pca, variance_ratios


# Description:
#    (3/29/24 Weekly Doc density ditributions plot function)
#    This function plots combined density histograms for a set of statistical measures within a given dataset. 
#    It filters the data by unique classes and selected 'k' values, calculates the bin width using the Freedman-Diaconis rule, 
#    and generates density histograms for each class-statistic pair. Specific 'k' values of interest, [2, 50, 100, 150, 199], are highlighted 
#    with different colors.
# Accepts:
#    DataFrame df: the input data containing the classes, 'k' values, and statistical measures.
#    list stats: a list of strings representing the statistical measures to be plotted.
# Returns:
#    None: This function does not return anything as its primary purpose is to render plots. 
#    It shows the plots directly with plt.show() for each class-statistic pair

def plot_density_histograms(df, stats):
    classes = df['Class'].unique()
    k_values = list(range(2,200))
    select_values = [2, 50, 100, 150 ,199]  # Specific 'k' values to plot
    colors = ['darkblue', 'orangered', 'forestgreen', 'red', 'darkmagenta']  # Colors for different k-values

    for current_class in classes:
        for stat in stats:
            # Prepare data for Freedman-Diaconis bin width calculation
            combined_data = df[(df['Class'] == current_class) & (df['k'].isin(k_values))][stat].dropna()
            iqr = np.subtract(*np.percentile(combined_data, [75, 25]))
            bin_width = 2 * iqr * (len(combined_data) ** (-1/3))
            bins = int((combined_data.max() - combined_data.min()) / bin_width)

            # Determine global bounds across 'k' values for comparison
            x_min = combined_data.min()
            x_max = combined_data.max()
            y_max = 0

            # Initialize plot
            plt.figure(figsize=(10, 6))
            plt.suptitle(f'Class {current_class}: Combined Density Histograms for {stat}', fontsize=16)

            counter = 0
            # Plot combined histograms
            for idx, k in enumerate(k_values):
                data_points = df[(df['k'] == k)][stat].dropna()
                hist, _ = np.histogram(data_points, bins=bins, density=True)
                y_max = max(y_max, max(hist))

                if k in select_values:
                  sns.histplot(data_points, bins=bins, kde=False, stat='density', color=colors[counter] , alpha=(0.2*counter)+0.1, label=f'k={k}')
                  counter = counter + 1
                    
            # Improve layout
            plt.xlim(x_min, x_max)
            plt.ylim(0, y_max)
            plt.xlabel(f'{stat} Value')
            plt.ylabel('Density')
            plt.grid(alpha=0.2)
            plt.legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

# Description:
#    (3/29/24 Weekly Doc Case-by-Case PCA plot function)
#    This function iteratively generates 2D PCA plots for specified replicates and classes within df.
#    It first determines global bounds for principal components to standardize plot ranges across different classes and replicates.
#    In the second pass, it performs PCA for each class and replicate pair, standardizing the data excluding specified columns. 
#    Plots display PCA scatter points, highlighting points with 'k' values in the select_k list and annotating them. 
# Accepts:
#    DataFrame df: the input dataset containing features along with 'Class' and 'Replicate' columns.
#    list exclude_columns: column names to be excluded from PCA feature set.
#    list rep_values: specific replicate values to generate plots for.
#    list select_k: specific 'k' values within the data to be highlighted and annotated in the plot.
# Returns:
#    None: This function does not return any value, as it is designed to generate and display plots directly.
def plot_pca_plots(df, exclude_columns, rep_values, select_k):
    classes = df['Class'].unique()

    # Exclude specified columns from PCA analysis
    features = [col for col in df.columns if col not in exclude_columns]

    global_min_pc1, global_max_pc1 = float('inf'), float('-inf')
    global_min_pc2, global_max_pc2 = float('inf'), float('-inf')

    # First pass: determine global PCA bounds
    for Class in classes:
        class_data = df[df['Class'] == Class]

        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(class_data[features])

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(standardized_data)

        min_pc1, max_pc1 = pca_result[:, 0].min(), pca_result[:, 0].max()
        min_pc2, max_pc2 = pca_result[:, 1].min(), pca_result[:, 1].max()

        global_min_pc1, global_max_pc1 = min(global_min_pc1, min_pc1), max(global_max_pc1, max_pc1)
        global_min_pc2, global_max_pc2 = min(global_min_pc2, min_pc2), max(global_max_pc2, max_pc2)

    # Add margins to the global bounds
    margin_pc1 = (global_max_pc1 - global_min_pc1) * 0.05
    margin_pc2 = (global_max_pc2 - global_min_pc2) * 0.05
    global_min_pc1 -= margin_pc1
    global_max_pc1 += margin_pc1
    global_min_pc2 -= margin_pc2
    global_max_pc2 += margin_pc2

    # Second pass: plot PCA evolution for each class with fixed bounds
    for Class in classes:
        for rep in rep_values:
            rep_data = df[(df['Class'] == Class) & (df['Replicate'] == rep)]
            rep_data_sorted = rep_data.sort_values(by='k')

            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(rep_data_sorted[features])

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(standardized_data)

            plt.figure(figsize=(10, 6))
            plt.title(f'2D PCA for Class {Class}, Replicate {rep}')

            for i in range(pca_result.shape[0]):
                k_value = rep_data_sorted.iloc[i]['k']
                color = 'dodgerblue' if k_value not in select_k else 'red'  # Highlight the last point with 'k' = 198
                alpha = 0.2 if k_value not in select_k else 0.8
                plt.scatter(pca_result[i, 0], pca_result[i, 1], color=color, alpha=alpha)

                # Only notate points if their 'k' value is in select_k
                if k_value in select_k:
                    plt.text(pca_result[i, 0], pca_result[i, 1], f"k={k_value}", fontsize=8)

            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.xlim(global_min_pc1, global_max_pc1)
            plt.ylim(global_min_pc2, global_max_pc2)
            plt.grid(alpha=0.2)
            plt.show()
