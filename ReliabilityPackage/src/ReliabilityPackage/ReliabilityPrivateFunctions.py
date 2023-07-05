import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, matthews_corrcoef, f1_score, recall_score,\
    brier_score_loss, mean_squared_error, balanced_accuracy_score
from ReliabilityPackage.ReliabilityClasses import DensityPrincipleDetector
import random
from collections import Counter


def _train_one_epoch(epoch_index, training_set, training_loader, optimizer, loss_function, ae):
    """
    Trains the autoencoder model for one epoch using the provided training set and loader.

    This function trains the autoencoder model for one epoch using the provided training set and data loader.
    It updates the model parameters, and calculates the training loss.

    :param int epoch_index: The index of the current epoch.
    :param numpy.ndarray training_set: The training set.
    :param torch.utils.data.DataLoader training_loader: The data loader for the training set.
    :param torch.optim.Optimizer optimizer: The optimizer used for parameter updates.
    :param torch.nn.Module loss_function: The loss function used for training.
    :param torch.nn.Module ae: The autoencoder model.

    :return: The average training loss per batch in the last epoch.
    :rtype: float
    """
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(training_loader):
        inputs = data
        optimizer.zero_grad()
        outputs = ae(inputs.float())
        loss = loss_function(outputs, inputs.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_set) + i + 1
            print('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def _compute_synpts_accuracy(predict_func, synpts, X_train, y_train, k=5):
    """
    Computes the accuracy of the synthetic points with the classifer.

    This function computes the accuracy on the set of synthetic points.
    The accuracy of each synthetic points is computed by comparing the predicted labels of its k nearest
    training samples to their actual labels.

    :param callable predict_func: The predict function of the classifier.
    :param numpy.ndarray synpts: The synthetic points with shape (n_synpts, n_features).
    :param numpy.ndarray X_train: The training data with shape (n_samples, n_features).
    :param numpy.ndarray y_train: The training labels with shape (n_samples,).
    :param int k: The number of nearest neighbors to consider (default: 5).

    :return: The accuracy scores associated with each synthetic point.
    :rtype: numpy.ndarray
    """
    acc_syn = []

    for i in range(len(synpts)):
        distances = np.linalg.norm(X_train - synpts[i], axis=1)
        nn = distances.argsort()[:k]
        acc_syn.append(accuracy_score(predict_func(X_train[nn, :]), y_train[nn]))
    acc_syn = np.asarray(acc_syn)

    return acc_syn


def _compute_metrics(y, ypred):
    """
    Computes various classification metrics based on the predicted and true labels.

    This function computes several classification metrics based on the predicted
    labels `ypred` and the true labels `y`. The metrics computed include accuracy,
    precision, recall, F1-score, Matthews correlation coefficient, and Brier score loss.

    :param 1d array-like y: The true labels.
    :param 1d array-like ypred: The predicted labels.

    :return: A list containing the computed metrics in the following order:
              [balanced_accuracy, precision, recall, F1-score, Matthews correlation coefficient, Brier score loss].
    :rtype: list
    """
    scores = [balanced_accuracy_score(y, ypred), precision_score(y, ypred), recall_score(y, ypred), f1_score(y, ypred),
              matthews_corrcoef(y, ypred), brier_score_loss(y, ypred)]

    return scores


def _dataset_density_reliability(X, density_principle_predictor):
    """
    Computes the density reliability of each data point in the dataset based on a density principle predictor.

    This function computes the density reliability of each data point in the dataset `X` based on a density
    principle predictor. It applies the `compute_reliability` function from the `density_principle_predictor`
    to each data point along axis 1 of `X`.

    :param array-like X: The input dataset.
    :param ReliabilityClasses.DensityPrinciplePredictor density_principle_predictor: An instance of a density principle
        predictor with a `compute_reliability` method.

    :return: An array containing the density reliability scores for each data point in `X`.
    :rtype: numpy.ndarray
    """
    return np.apply_along_axis(density_principle_predictor.compute_reliability, 1, X)


def _find_first_projection(x_i, autoencoder):
    """
    Computes the first projection of an autoencoder for a given input.

    This function computes the first projection of the `autoencoder` for a given input `x_i`. It first converts
    the input `x_i` into a tensor, passes it through the `autoencoder`, and then returns the projection as a numpy array.

    :param array-like x_i: The input data for which to compute the projection.
    :param torch.nn.Module autoencoder: The autoencoder used for projection.

    :return: The first projection of the `autoencoder` for the input `x_i` as a numpy array.
    :rtype: numpy.ndarray
    """
    pred = autoencoder((torch.tensor(x_i)).float())
    return pred.detach().numpy()


def _dataset_first_projections(X, autoencoder):
    """
    Computes the first projections of an autoencoder for each data point in a dataset.

    This function computes the first projection of the `autoencoder` for each data point in the dataset `X`.
    It applies the `find_first_projection` function along the axis 1 of `X` to compute the projections for each data point.

    :param array-like X: The dataset for which to compute the first projections.
    :param atorch.nn.Module autoencoder: The autoencoder used for projection.

    :return: An array containing the first projections of the `autoencoder` for each data point in `X`.
    :rtype: numpy.ndarray
    """
    return np.apply_along_axis(_find_first_projection, 1, X, autoencoder)


def _val_scores_diff_mse(ae, X_val, y_val, predict_func):
    """
    Computes different scores on the reliable and unreliable subset of the validation set based on different mean squared error (MSE) thresholds.

    This function computes different scores on the reliable and unreliable subset of the validation set based on different mean squared error (MSE) thresholds.
    It calculates the MSE for each data point in `X_val` and then generates a list of MSE thresholds using percentiles.
    For each threshold, it computes different scores for the reliable and unreliable samples obtained, and the number and percentage of unreliable samples.
    Finally, it returns the MSE threshold list and, for each threshold, the scores computed on the reliable and unreliable samples
    and the number and percentage of unreliable samples.

    :param torch.nn.Module ae: The autoencoder used for projection.
    :param array-like X_val: The validation dataset.
    :param array-like y_val: The validation labels.
    :param callable predict_func: The predict function of the classifier.

    :return: A tuple containing the MSE threshold list, and, for each threshold, the reliability scores, the unreliable scores,
          the number of unreliable samples and the percentage of unreliable samples
    :rtype: tuple
    """
    mse_val = []
    for i in range(len(X_val)):
        val_projections = _dataset_first_projections(X_val, ae)
        mse_val.append(mean_squared_error(X_val[i], val_projections[i]))

    mse_threshold_list = [np.percentile(mse_val, i) for i in range(2, 100)]

    rel_scores = []
    unrel_scores = []
    perc_unrel = []
    num_unrel = []
    for i in range(len(mse_threshold_list)):
        print('iterata', i + 1, "/", len(mse_threshold_list))
        DP = DensityPrincipleDetector(ae, mse_threshold_list[i])
        val_reliability = _dataset_density_reliability(X_val, DP)
        reliable_val = X_val[np.where(val_reliability == 1)]
        unreliable_val = X_val[np.where(val_reliability == 0)]
        y_reliable_val = y_val[np.where(val_reliability == 1)]
        y_unreliable_val = y_val[np.where(val_reliability == 0)]
        ypred_reliable_val = predict_func(reliable_val)
        ypred_unreliable_val = predict_func(unreliable_val)
        rel_scores.append(_compute_metrics(y_reliable_val, ypred_reliable_val))
        unrel_scores.append(_compute_metrics(y_unreliable_val, ypred_unreliable_val))
        num_unrel.append((len(X_val) - sum(val_reliability)))
        perc_unrel.append((len(X_val) - sum(val_reliability)) / len(X_val))

    return mse_threshold_list, rel_scores, unrel_scores, num_unrel, perc_unrel


def _generate_binary_vector(length):
    """
    Generates a binary vector of a specified length.

    This function generates a binary vector of a specified length, where each element in the vector is a randomly generated
    binary digit (0 or 1).

    :param int length: The desired length of the binary vector.

    :return: A list representing the generated binary vector.
    :rtype: list
    """
    binary_vector = []
    for _ in range(length):
        random_bit = random.randint(0, 1)
        binary_vector.append(random_bit)
    return binary_vector


def _contains_only_integers(array):
    """
    Checks if an array contains only integer values.

    :param array-like array: The array to be checked.

    :return: True if the array contains only integer values, False otherwise.
    :rtype: bool
    """
    integers_array = np.asarray(array).astype(int)
    check_array = np.unique(integers_array == array)
    if len(check_array) == 1 and check_array[0]:
        return True
    else:
        return False


def _extract_values_proportionally(array):
    """
    Extracts values from an array proportionally to their frequencies.

    :param array-like array: The array containing values.

    :return: A list of values extracted proportionally to their frequencies.
    :rtype: list
    """
    extracted_array = []
    frequencies = Counter(array)
    total_count = sum(frequencies.values())
    proportions = {value: count / total_count for value, count in frequencies.items()}
    for i in range(len(array)):
        extracted_array.append(random.choices(list(proportions.keys()), list(proportions.values()))[0])

    return extracted_array
