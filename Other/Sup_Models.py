import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

# Dictionary to map the len_features to chunk_size/group value
GROUP_MAP = {
    5346: 27,
    1782: 9,
    5544: 28,
    1980: 10
}

def process_dataframe(df):
    # Extract the 'Class' column
    class_column = df['Class']

    # One-hot encode the 'Class' column
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(class_column.values.reshape(-1, 1))

    # Drop the 'Class' and 'Replicate' columns from the original DataFrame
    df = df.drop(columns=['Class', 'Replicate'])

    # Convert nested arrays to flat arrays and ensure uniform shape
    flattened_data = np.array([np.concatenate([np.ravel(cell) for cell in row]) for row in df.values])

    return flattened_data, Y

def display_important_features(model, feature_count, is_rf=False):
    if is_rf:
        # Get feature importances for RandomForest
        feature_importances = model.feature_importances_

        # Sort feature importances in descending order and get the indices of the top 5
        important_features_indices = np.argsort(feature_importances)[::-1][:5]

        # Calculate the influence percentage
        total_importance = np.sum(feature_importances)
        influence_percentage = (feature_importances / total_importance) * 100

        # Map feature indices to names (assuming feature indices are the names for now)
        mapped_feature_names = map_array_elements(list(range(feature_count)), feature_count)

    else:
        # Handle linear models with coef_ attribute
        coefficients = model.coef_
        importance = np.mean(np.abs(coefficients), axis=0)
        
        # Calculate influence percentage
        influence_percentage = (importance / np.sum(importance)) * 100
        important_features_indices = np.argsort(influence_percentage)[::-1][:5]
        
        # Map feature indices to names
        mapped_feature_names = map_array_elements(list(range(feature_count)), feature_count)
    
    # Display the top 5 most important features
    print("\nTop 5 Most Important Features and Their Influence Percentage:")
    for idx in important_features_indices[:5]:
        print(f"{mapped_feature_names[idx]}, Influence Percentage: {influence_percentage[idx]:.2f}%")


# Function to calculate and display the confusion matrix
def display_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

# SVM Model
def svm_model(X_train, Y_train, X_test, Y_test):
    # Convert one-hot encoded labels back to single labels
    Y_train_single = np.argmax(Y_train, axis=1)
    Y_test_single = np.argmax(Y_test, axis=1)

    def train_and_evaluate_svm(kernel, degree=None):
        if kernel == 'poly':
            model = SVC(kernel=kernel, degree=degree)
        else:
            model = SVC(kernel=kernel)

        model.fit(X_train, Y_train_single)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test_single, y_pred)
        return model, accuracy, y_pred

    # Train and evaluate SVM with linear kernel
    linear_model, linear_accuracy, linear_y_pred = train_and_evaluate_svm('linear')
    print(f'Accuracy (Linear Kernel): {linear_accuracy * 100:.2f}%')
    display_important_features(linear_model, X_train.shape[1])
    display_confusion_matrix(Y_test_single, linear_y_pred)
    print('\n--------------------\n')

    # Train and evaluate SVM with RBF kernel
    rbf_model, rbf_accuracy, rbf_y_pred = train_and_evaluate_svm('rbf')
    print(f'Accuracy (RBF Kernel): {rbf_accuracy * 100:.2f}%')
    display_confusion_matrix(Y_test_single, rbf_y_pred)
    print('\n--------------------\n')

    # Train and evaluate SVM with poly kernel (degree 2)
    poly_model_deg2, poly_accuracy_deg2, poly_y_pred_deg2 = train_and_evaluate_svm('poly', degree=2)
    print(f'Accuracy (Poly Kernel, degree=2): {poly_accuracy_deg2 * 100:.2f}%')
    display_confusion_matrix(Y_test_single, poly_y_pred_deg2)
    print('\n--------------------\n')

    # Train and evaluate SVM with poly kernel (degree 3)
    poly_model_deg3, poly_accuracy_deg3, poly_y_pred_deg3 = train_and_evaluate_svm('poly', degree=3)
    print(f'Accuracy (Poly Kernel, degree=3): {poly_accuracy_deg3 * 100:.2f}%')
    display_confusion_matrix(Y_test_single, poly_y_pred_deg3)

# Random Forest Model
def rf_model(X_train, Y_train, X_test, Y_test, n_estimators=200):
    # Convert Y_train and Y_test from one-hot to single-label format
    Y_train_single_label = np.argmax(Y_train, axis=1)
    Y_test_single_label = np.argmax(Y_test, axis=1)

    # Train the RandomForest model
    rf_model = RandomForestClassifier(n_estimators=n_estimators)
    rf_model.fit(X_train, Y_train_single_label)

    # Predict the test set
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(Y_test_single_label, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Display the most important features
    display_important_features(rf_model, X_train.shape[1], is_rf=True)


# Logistic Regression Model
def logistic_regression(X_train, Y_train, X_test, Y_test):
    Y_train_single = np.argmax(Y_train, axis=1)
    Y_test_single = np.argmax(Y_test, axis=1)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000)
    model.fit(X_train, Y_train_single)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test_single, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Display the most important features
    display_important_features(model, X_train.shape[1])
    
    # Display the confusion matrix
    display_confusion_matrix(Y_test_single, y_pred)

# Main function to handle both balanced and raw datasets
def binary_logistic_regression(df):
    # Create binary labels for the whole dataframe
    binary_class = df['Class'].apply(lambda x: 0 if x == 'E' else 1)
    # Separate features and target
    X = np.array(df['Vector'].tolist())
    y = binary_class

    def train_and_evaluate(X, y):
        # Split into an 80/20 training/test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the logistic classifier on the training set
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Output the accuracy on the test set
        accuracy = accuracy_score(y_test, y_pred)
        print(f'\nAccuracy: {accuracy * 100:.2f}%')
        
        # Display the top 10 most important features
        display_important_features(model, X_train.shape[1])
        
        # Display the confusion matrix
        display_confusion_matrix(y_test, y_pred)
        
        return accuracy

    # Case 1: Balanced dataset
    num_e_rows = len(df[df['Class'] == 'E'])
    other_classes = df[df['Class'] != 'E']['Class'].unique()
    num_other_samples = num_e_rows // len(other_classes)

    sampled_dfs = [df[df['Class'] == cls].sample(num_other_samples, random_state=42) for cls in other_classes]
    sampled_dfs.append(df[df['Class'] == 'E'])
    balanced_df = pd.concat(sampled_dfs)

    X_balanced = np.array(balanced_df['Vector'].tolist())
    y_balanced = balanced_df['Class'].apply(lambda x: 0 if x == 'E' else 1)

    print("Balanced dataset evaluation:")
    balanced_accuracy = train_and_evaluate(X_balanced, y_balanced)
    print(f'\nAccuracy on balanced dataset: {balanced_accuracy * 100:.2f}%')

    # Case 2: Raw dataset
    print("\nRaw dataset evaluation:")
    raw_accuracy = train_and_evaluate(X, y)
    print(f'\nAccuracy on raw dataset: {raw_accuracy * 100:.2f}%')

def map_string(s):
    if s.endswith('_1'):
        return f"mean of {s[:-2]}"
    elif s.endswith('_2'):
        return f"mean of {s[:-2]}"
    elif s.endswith('_3'):
        return f"mean of {s[:-2]}"

def map_array_elements(array, len_features):
    int_mapping_all = [
        'k={}', 'mean of alpha_1 for k={}', 'var of alpha_1 for k={}', 'SE of alpha_1 for k={}',
        'mean of alpha_2 for k={}', 'var of alpha_2 for k={}', 'SE of alpha_2 for k={}',
        'mean of alpha_3 for k={}', 'var of alpha_3 for k={}', 'SE of alpha_3 for k={}',
        'mean of pi_1 for k={}', 'var of pi_1 for k={}', 'SE of pi_1 for k={}',
        'mean of pi_2 for k={}', 'var of pi_2 for k={}', 'SE of pi_2 for k={}',
        'mean of pi_3 for k={}', 'var of pi_3 for k={}', 'SE of pi_3 for k={}',
        'mean of pihat_12 for k={}', 'var of pihat_12 for k={}', 'SE of pihat_12 for k={}',
        'mean of pihat_13 for k={}', 'var of pihat_13 for k={}', 'SE of pihat_13 for k={}',
        'mean of pihat_23 for k={}', 'var of pihat_23 for k={}', 'SE of pihat_23 for k={}'
    ]

    group = GROUP_MAP.get(len_features, 1)
    
    if group == 27:
        mapping = int_mapping_all[1:]  # Keep all but 'k={}'
    elif group == 9:
        mapping = [item for item in int_mapping_all if 'mean of' in item]  # Keep only 'mean of' strings
    elif group == 28:
        mapping = int_mapping_all  # Keep all elements
    elif group == 10:
        mapping = [item for item in int_mapping_all if 'mean of' in item or item == 'k={}']  # Keep 'mean of' and 'k={}'
    else:
        print('Number of features:',len_features)
        return np.array([map_string(s) for s in array])  # Fallback for len_features <= 100

    result = [
        mapping[k % group].format(k // group + 2)
        for k in array
    ]
    return np.array(result)

