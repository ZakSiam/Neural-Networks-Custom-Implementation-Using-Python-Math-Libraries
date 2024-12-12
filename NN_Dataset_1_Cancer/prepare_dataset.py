import numpy as np
import pandas as pd

# Load the Breast Cancer dataset
data = pd.read_csv('breast-cancer.data', header=None, na_values='?')

# Rename columns for clarity (based on UCI repository description)
data.columns = [
    'Class', 'Age', 'Menopause', 'TumorSize', 'InvNodes',
    'NodeCaps', 'DegMalig', 'Breast', 'BreastQuad', 'Irradiat'
]

# Drop rows with missing values
data.dropna(inplace=True)

# Separate features and target
X = data.iloc[:, 1:]  # Feature columns
y = data['Class']  # Target column

# Encode target variable (0 for "no-recurrence-events", 1 for "recurrence-events")
y = (y == 'recurrence-events').astype(int).values

# Convert categorical features into numerical values using unique integer mapping
def encode_categorical_column(column):
    unique_values = column.unique()
    value_to_int = {value: idx for idx, value in enumerate(unique_values)}
    return column.map(value_to_int)

for col in ['Age', 'Menopause', 'TumorSize', 'InvNodes', 'NodeCaps', 'Breast', 'BreastQuad', 'Irradiat']:
    X[col] = encode_categorical_column(X[col])

# Convert X to numpy array
X = X.values.astype(float)

# Normalize features (mean = 0, std = 1)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Split into train and test sets (80% train, 20% test)
def train_test_split_manual(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    test_split_index = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:test_split_index]
    test_indices = indices[test_split_index:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2)

# Oversample minority class
def oversample_minority_class(X, y):
    unique, counts = np.unique(y, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]

    # Extract minority class samples
    X_minority = X[y == minority_class]
    y_minority = y[y == minority_class]

    # Calculate the number of samples to add
    num_samples_to_add = counts[np.argmax(counts)] - counts[np.argmin(counts)]

    # Randomly sample with replacement
    indices = np.random.choice(len(X_minority), num_samples_to_add, replace=True)
    X_oversampled = np.vstack([X, X_minority[indices]])
    y_oversampled = np.hstack([y, y_minority[indices]])

    return X_oversampled, y_oversampled

# Apply oversampling to both training and test sets
X_train, y_train = oversample_minority_class(X_train, y_train)
X_test, y_test = oversample_minority_class(X_test, y_test)

# Save datasets as .npy files
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("\nBalanced datasets saved as .npy files!")
