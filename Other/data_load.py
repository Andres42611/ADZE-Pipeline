import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Description:
#   Concatenates multiple CSV files from different cases into a single DataFrame
#   Reads CSV files from specified paths, combines them, and randomly shuffles the rows
# Accepts:
#   str source: Base directory path containing the case folders
# Returns:
#   pandas.DataFrame: Combined and shuffled DataFrame containing data from all cases
def concat_csv(source):
    class_files = [
        source + '/CaseA/caseA_data.csv',
        source + '/CaseB/caseB_data.csv',
        source + '/CaseC/caseC_data.csv',
        source + '/CaseD/caseD_data.csv',
        source + '/CaseE/caseE_data.csv'
    ]
    dataframes = []

    for file in class_files:
        df_temp = pd.read_csv(file)
        dataframes.append(df_temp)

    df = pd.concat(dataframes, ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    return df

# Description:
#   Transforms a DataFrame of statistical data into a dictionary of vectors
#   Each vector represents a unique combination of class and replicate and provides statistics
#   from k=2 to k=199
# Accepts:
#   pandas.DataFrame df: DataFrame containing statistical data with columns for Class, Replicate, k, and various statistics
# Returns:
#   dict: Dictionary where keys are 'Class_Replicate' and values are flattened vectors of statistical data
def replicate_vector_dict(df):
    fin_dict = {}
    stat_col_names = [col for col in df.columns if col not in ['Class', 'Replicate', 'k']]
    grouped = df.groupby(['Class', 'Replicate'])

    for (cls, rep), group in grouped:
        possible_k = group['k'].unique()
        assert set(possible_k) == set(range(2, 200)), f"Missing k values for class {cls} and replicate {rep}"

        stat_data = group[stat_col_names].values
        fin_v = stat_data.flatten().tolist()
        fin_dict[f"{cls}_{rep}"] = fin_v

    return fin_dict

# Description:
#   Expands columns containing multiple values (mean, variance, standard error) into separate columns
#   Processes a DataFrame to split object-type columns (except 'Class') into individual numeric columns
# Accepts:
#   pandas.DataFrame df: Input DataFrame with columns potentially containing multiple values as strings
# Returns:
#   pandas.DataFrame: Expanded DataFrame with separate columns for each value
def full_df(df):
    for column in df.columns:
        if df[column].dtype == object and column not in ['Class', 'k', 'Replicate']:
            expanded_columns = df[column].str.strip('()').str.split(',', expand=True)
            expanded_columns = expanded_columns.astype(float)
            expanded_column_names = [f"{column}_{i+1}" for i in range(expanded_columns.shape[1])]
            df[expanded_column_names] = expanded_columns
            df.drop(column, axis=1, inplace=True)

    return df

def downsample_by_class(df, n_samples=100):
    downsampled_df = df.groupby('Class').sample(n=n_samples, random_state=24)
    return downsampled_df

# Description:
#   Selectively drops columns from a DataFrame based on statistical moment type
#   Allows for removal of mean, variance, and/or standard error columns
# Accepts:
#   pandas.DataFrame df: Input DataFrame with columns for different statistical moments
#   bool mu: If True, drops columns containing mean values (default: False)
#   bool variance: If True, drops columns containing variance values (default: False)
#   bool se: If True, drops columns containing standard error values (default: False)
# Returns:
#   pandas.DataFrame: DataFrame with specified moment columns removed
def drop_moments(df, mu=False, variance=False, se=False):
    drop_map = {'1': mu, '2': variance, '3': se}
    columns_to_drop = [col for col in df.columns if drop_map.get(col[-1], False)]
    df_raw = df.drop(columns=columns_to_drop)
    return df_raw

def vectorize_df(df_raw):
    vec_dict = replicate_vector_dict(df_raw)
    df_vec = pd.DataFrame([(key.split('_')[0], key.split('_')[1], value) for key, value in vec_dict.items()],
                          columns=['Class', 'Replicate', 'Vector'])
    df_vec = df_vec.sample(frac=1).reset_index(drop=True)
    return df_vec

def normalize_vector(vectors):
    return [list(np.array(vector) / np.linalg.norm(vector)) for vector in vectors]

def standardize_repvec_df(df, L2=False):
    float_vectors = df['Vector'].tolist()

    if not L2:
        SS_df = df.copy()
        scaler = StandardScaler()
        standardized_vectors = scaler.fit_transform(float_vectors)
        SS_df['Vector'] = standardized_vectors.tolist()
        return SS_df
    else:
        L2_df = df.copy()
        normalized_vectors = normalize_vector(float_vectors)
        L2_df['Vector'] = normalized_vectors
        return L2_df

def standardize_full_df(df, L2=False):
    feature_cols = [col for col in df.columns if col not in ['Class', 'Replicate']]
    vectors = df[feature_cols].values.astype(float)

    if not L2:
        SS_df = df.copy()
        scaler = StandardScaler()
        standardized_vectors = scaler.fit_transform(vectors)
        SS_df[feature_cols] = standardized_vectors
        return SS_df
    else:
        L2_df = df.copy()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_vectors = vectors / norms
        L2_df[feature_cols] = normalized_vectors
        return L2_df

# Description:
#   Generates various types of DataFrames based on a given numeric code (map below)
#   Combines CSV files, applies different transformations and standardizations
# Accepts:
#   str source: Path to the source directory containing CSV files
#   int code: Code (1-12) specifying which type of DataFrame to generate
# Returns:
#   pandas.DataFrame: Generated DataFrame based on the specified code
def make_df(source, code):
    comb_df = concat_csv(source)

    def generate_raw_df():
        raw_df = full_df(comb_df)
        return raw_df

    def generate_repvec_df(raw_df=None):
        if raw_df is None:
            raw_df = generate_raw_df()
        repvec_df = vectorize_df(raw_df)
        del raw_df
        return repvec_df

    def generate_raw_df_mu():
        raw_df = generate_raw_df()
        raw_df_mu = drop_moments(raw_df, mu=False, variance=True, se=True)
        del raw_df
        return raw_df_mu

    def generate_repvec_df_mu():
        raw_df_mu = generate_raw_df_mu()
        repvec_df_mu = vectorize_df(raw_df_mu)
        del raw_df_mu
        return repvec_df_mu

    def generate_repvec_df_SS():
        repvec_df = generate_repvec_df()
        repvec_df_SS = standardize_repvec_df(repvec_df)
        del repvec_df
        return repvec_df_SS

    def generate_repvec_df_L2():
        repvec_df = generate_repvec_df()
        repvec_df_L2 = standardize_repvec_df(repvec_df, L2=True)
        del repvec_df
        return repvec_df_L2

    def generate_repvec_df_mu_SS():
        repvec_df_mu = generate_repvec_df_mu()
        repvec_df_mu_SS = standardize_repvec_df(repvec_df_mu)
        del repvec_df_mu
        return repvec_df_mu_SS

    def generate_repvec_df_mu_L2():
        repvec_df_mu = generate_repvec_df_mu()
        repvec_df_mu_L2 = standardize_repvec_df(repvec_df_mu, L2=True)
        del repvec_df_mu
        return repvec_df_mu_L2

    def generate_raw_df_SS():
        raw_df = generate_raw_df()
        raw_df_SS = standardize_full_df(raw_df)
        del raw_df
        return raw_df_SS

    def generate_raw_df_L2():
        raw_df = generate_raw_df()
        raw_df_L2 = standardize_full_df(raw_df, L2=True)
        del raw_df
        return raw_df_L2

    def generate_raw_df_mu_SS():
        raw_df_mu = generate_raw_df_mu()
        raw_df_mu_SS = standardize_full_df(raw_df_mu)
        del raw_df_mu
        return raw_df_mu_SS

    def generate_raw_df_mu_L2():
        raw_df_mu = generate_raw_df_mu()
        raw_df_mu_L2 = standardize_full_df(raw_df_mu, L2=True)
        del raw_df_mu
        return raw_df_mu_L2

    code_to_function = {
        1: generate_raw_df,
        2: generate_repvec_df,
        3: generate_raw_df_mu,
        4: generate_repvec_df_mu,
        5: generate_repvec_df_SS,
        6: generate_repvec_df_L2,
        7: generate_repvec_df_mu_SS,
        8: generate_repvec_df_mu_L2,
        9: generate_raw_df_SS,
        10: generate_raw_df_L2,
        11: generate_raw_df_mu_SS,
        12: generate_raw_df_mu_L2,
    }

    if code in code_to_function:
        df = code_to_function[code]()
        print("DataFrame Info:")
        print(df.info())
        print("DataFrame Head:")
        print(df.head())
        return df
    else:
        raise ValueError("Invalid code. Please enter a number between 1 and 12.")
