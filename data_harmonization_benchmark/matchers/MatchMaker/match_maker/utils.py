
from .constants import NULL_REPRESENTATIONS, BINARY_VALUES, KEY_REPRESENTATIONS
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from valentine import MatcherResults


def convert_to_valentine_format(matched_columns, source_table, target_table):
    valentine_format = {}
    for source_column, matches in matched_columns.items():
        for target_column, score in matches:
            key = (source_table, source_column), (target_table, target_column)
            valentine_format[key] = score
    if isinstance(valentine_format, MatcherResults):
        return valentine_format
    return MatcherResults(valentine_format)

def common_prefix(strings):
    if not strings:
        return ""

    # Sort the list, the common prefix of the whole list would be the common prefix of the first and last string
    strings.sort()
    first = strings[0]
    last = strings[-1]

    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1

    return first[:i]

def common_ngrams(strings, threshold=0.3):
    most_common_ngrams = {}

    # Loop through n-gram sizes from 3 to 8
    for n in range(3, 9):
        # Create a TfidfVectorizer for n-grams of size 'n'
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n, n))
        
        # Fit and transform the list of strings
        tfidf_matrix = vectorizer.fit_transform(strings)
        
        # Sum the tf-idf scores across all documents for each n-gram
        scores = tfidf_matrix.sum(axis=0)
        
        # Get feature names (n-grams) and their corresponding scores
        ngram_scores = [(ngram, scores[0, idx]) for ngram, idx in vectorizer.vocabulary_.items()]
        
        # Filter n-grams by threshold score
        filtered_ngrams = [ngram for ngram in ngram_scores if ngram[1] > threshold]
        
        # Sort n-grams by score in descending order
        most_common_ngrams[n] = sorted(filtered_ngrams, key=lambda x: x[1], reverse=True)
    
    # Return the filtered and sorted n-grams
    return most_common_ngrams



def preprocess_string(s):
    # Remove non-alphanumeric characters and convert to lowercase
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()


def alignment_score(str1, str2):
    # Preprocess strings
    str1 = preprocess_string(str1)
    str2 = preprocess_string(str2)

    # Determine shorter and longer strings
    if len(str1) <= len(str2):
        shorter, longer = str1, str2
    else:
        shorter, longer = str2, str1

    matches = 0
    last_index = -1

    # Find matches for each letter in the shorter string
    for char in shorter:
        for i in range(last_index + 1, len(longer)):
            if longer[i] == char:
                matches += 1
                last_index = i
                break

    # Calculate score
    score = matches / len(shorter)
    return score








def is_null_value(value):
    if isinstance(value, str):
        value = value.lower()
    return value in NULL_REPRESENTATIONS


def is_binary_value(value):
    if isinstance(value, str):
        value = value.lower()
    return value in BINARY_VALUES


def remove_invalid_characters(input_string):
    # Remove any character that is not a letter, digit, or whitespace
    pattern = r'[^a-zA-Z0-9\s]'
    cleaned_string = re.sub(pattern, ' ', input_string)
    return cleaned_string


def split_camel_case(input_string):
    # Split camel case by adding a space before any uppercase letter that is followed by a lowercase letter
    split_string = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', input_string)
    return split_string


def clean_column_name(col_name):
    # Strip leading/trailing spaces, convert to lowercase, split camel case, and remove invalid characters
    col_name = col_name.strip()
    col_name = split_camel_case(col_name)
    col_name = col_name.lower()
    col_name = remove_invalid_characters(col_name)
    # Reduce multiple spaces to a single space
    col_name = re.sub(r'\s+', ' ', col_name)
    return col_name


def clean_df(df):

    def clean_element(x):
        if is_null_value(x):
            return None
        if isinstance(x, str):
            val = split_camel_case(x)
            val = remove_invalid_characters(val.strip().lower())
            
            if val != "":
                return val
            else:
                return None
        return x

    df = df.apply(lambda col: col.apply(clean_element))

    return df


def detect_column_type(col, key_threshold=0.8, numeric_threshold=0.90):

    if "gene" in col.name.lower():
        # TODO, implement a less naive approach
        return "gene"

    if "date" in col.name.lower():
        # TODO, implement a less naive approach
        return "date"
    
    

    unique_values = col.dropna().unique()
    if len(unique_values)/len(col) > key_threshold and col.dtype not in [np.float64, np.float32, np.float16]:
        # columns with many distinct values are considered as "keys"
        return "key"
    

    if len(unique_values) == 0:
        return "Unknown"

    
    col_name = col.name.lower()
    if any(col_name.startswith(rep) or col_name.endswith(rep) for rep in KEY_REPRESENTATIONS):
        return "key"

    if col.dtype in [np.float64, np.int64] :
        return "numerical"

    numeric_unique_values = pd.Series(
        pd.to_numeric(unique_values, errors='coerce'))
    numeric_unique_values = numeric_unique_values.dropna()

    if not numeric_unique_values.empty:
        if len(numeric_unique_values) / len(unique_values) > numeric_threshold:
            if len(numeric_unique_values) > 2:
                return "numerical"
            else:
                unique_values_as_int = set(map(int, unique_values))
                if unique_values_as_int.issubset({0, 1}):
                    return "binary"
                else:
                    return "numerical"

    if len(unique_values) == 2 and all(is_binary_value(val) for val in unique_values):
        return "binary"
    else:
        return "categorical"

    raise ValueError(f"Could not detect type for column {col.name}")


def get_type2columns_map(df):
    # TODO: add more types, maybe semantic types
    types2columns_map = {}
    types2columns_map["key"] = []
    types2columns_map["numerical"] = []
    types2columns_map["categorical"] = []
    types2columns_map["binary"] = []
    types2columns_map["gene"] = []
    types2columns_map["date"] = []
    types2columns_map["Unknown"] = []

    for col in df.columns:
        col_type = detect_column_type(df[col])
        types2columns_map[col_type].append(col)

    return types2columns_map

# def get_samples(values, n=15, random=True):
#     unique_values = values.dropna()#.unique()
#     if random:
#         tokens = np.random.choice(
#             unique_values, min(15, len(unique_values)), replace=False
#         )
#     else:
#         value_counts = values.dropna().value_counts()
#         most_frequent_values = value_counts.head(n)
#         tokens = most_frequent_values.index.tolist()
#     return [str(token) for token in tokens]

def get_samples(values, n=10, random=True):
    unique_values = values.dropna().unique()
    total_unique = len(unique_values)

    # If total unique values are fewer than `n`, return them all sorted as strings
    if total_unique <= n:
        return sorted([str(val) for val in unique_values])

    if random:
        # Select n/2 most frequent values and n/2 evenly spaced unique values
        value_counts = values.dropna().value_counts()
        most_frequent_values = value_counts.head(n // 2).index.tolist()
        
        # For diversity, choose `n - len(most_frequent_values)` evenly spaced values
        spacing_interval = max(1, total_unique // (n - len(most_frequent_values)))
        diverse_values = unique_values[::spacing_interval][:n - len(most_frequent_values)]
        
        # Combine the frequent and diverse samples, remove duplicates, and sort them
        tokens = sorted(set(most_frequent_values + list(diverse_values)))
        
    else:
        # Deterministic approach: only take `n` most frequent values, sorted
        value_counts = values.dropna().value_counts()
        tokens = sorted(value_counts.head(n).index.tolist())

    return [str(token) for token in tokens]



