import numpy as np
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from joblib import Parallel, delayed

def generate_unsampled_indices(random_state, n_samples):
    """
    Generate the indices of the samples that were not included in the bootstrap sample.
    """
    rng = np.random.RandomState(random_state)
    sampled_indices = rng.randint(0, n_samples, n_samples)
    unsampled_mask = np.ones(n_samples, dtype=bool)
    unsampled_mask[sampled_indices] = False
    unsampled_indices = np.where(unsampled_mask)[0]
    return unsampled_indices

def calculate_rf_gap_proximity_matrix(rf, X):
    n_samples = X.shape[0]
    proximity_matrix = np.zeros((n_samples, n_samples))

    for tree in rf.estimators_:
        # Generate out-of-bag (OOB) indices manually
        oob_indices = generate_unsampled_indices(tree.random_state, n_samples)

        # Get the leaf indices for each sample
        leaf_indices = tree.apply(X)

        for i in range(n_samples):
            for j in range(n_samples):
                if leaf_indices[i] == leaf_indices[j]:
                    if i in oob_indices or j in oob_indices:
                        # Increase proximity if one of the samples is out-of-bag
                        proximity_matrix[i, j] += 1

    # Normalize by the number of trees to get the proximity score
    proximity_matrix /= len(rf.estimators_)

    return proximity_matrix
def weighted_percentile(values, quantiles, sample_weight=None):
    """
    Compute the weighted percentiles for a given array.

    :param values: 1D array of data
    :param quantiles: array-like with quantiles to compute, which should be in the range [0, 100]
    :param sample_weight: 1D array of sample weights
    :return: array of computed percentiles
    """
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)

    if sample_weight is None:
        sample_weight = np.ones(len(values))

    # Sort values and corresponding weights
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    # Compute the weighted quantiles
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles / 100.0, weighted_quantiles, values)

def rf_gap_quantile_regression(X_train, y_train, X_test, quantiles, rf):
    # Calculate the proximity matrix for the training data
    proximity_matrix = calculate_rf_gap_proximity_matrix(rf, X_train)
    predictions = np.zeros((X_test.shape[0], len(quantiles)))

    for i, x_test in enumerate(X_test):
        # Calculate proximity of the test point to each training point
        proximities = np.zeros(X_train.shape[0])
        for tree in rf.estimators_:
            leaf_index_test = tree.apply([x_test])[0]
            leaf_indices_train = tree.apply(X_train)
            proximities += (leaf_indices_train == leaf_index_test).astype(int)

        # Normalize proximities to be used as weights
        proximities /= len(rf.estimators_)

        # Calculate weighted percentiles
        for j, q in enumerate(quantiles):
            weighted_responses = weighted_percentile(y_train, [q], sample_weight=proximities)
            predictions[i, j] = weighted_responses[0]

    return predictions
def calculate_proximity_parallel(tree, X_train, x_test):
    leaf_index_test = tree.apply([x_test])[0]
    leaf_indices_train = tree.apply(X_train)
    return (leaf_indices_train == leaf_index_test).astype(int)

def rf_gap_quantile_regression_parallel(X_train, y_train, X_test, quantiles, rf):
    predictions = np.zeros((X_test.shape[0], len(quantiles)))

    for i, x_test in enumerate(X_test):
        # Initialize proximities as float array
        proximities = np.zeros(X_train.shape[0], dtype=float)

        # Parallel computation of proximities
        proximity_sums = Parallel(n_jobs=-1)(delayed(calculate_proximity_parallel)(tree, X_train, x_test) for tree in rf.estimators_)
        proximities += np.sum(proximity_sums, axis=0)

        # Normalize proximities
        proximities /= len(rf.estimators_)

        # Calculate weighted percentiles
        for j, q in enumerate(quantiles):
            weighted_responses = weighted_percentile(y_train, [q], sample_weight=proximities)
            predictions[i, j] = weighted_responses[0]

    return predictions


def qrf_predict(rf, X_test, quantiles):
    predictions = np.zeros((X_test.shape[0], len(quantiles)))

    for i, q in enumerate(quantiles):
        predictions[:, i] = np.percentile([tree.predict(X_test) for tree in rf.estimators_], q, axis=0)

    return predictions
