from get_data import get_data
from naive_bayes_classifier import NaiveBayesClassifier
from random import shuffle


def split_dataset_in_equal_subsets(dataset: list, k: int = 5):
    """
    Function splits data set into k subsets of almost identical size.

    param k: number of expected subsets
    type k: int
    """
    delta = round(len(dataset) / k)
    idx_prev = 0
    idx_curr = delta
    new_subsets = []
    for _ in range(k-1):
        new_subsets.append(dataset[idx_prev:idx_curr])
        idx_prev = idx_curr
        idx_curr += delta
    new_subsets.append(dataset[idx_prev:])
    return new_subsets


def k_fold_validation(dataset: list, k: int = 5):
    """
    Function perfomrs k-folded cross validation.

    param k: number of expected subsets
    type k: int
    """
    k_folded_dataset = split_dataset_in_equal_subsets(dataset, k)
    training_set = []
    test_set = []
    total_effectiveness = 0
    for i in range(k):
        training_set = []
        test_set = []
        for j in range(k):
            if j != i:
                training_set += k_folded_dataset[j]
        test_set = k_folded_dataset[i]
        nb_classifier = NaiveBayesClassifier()
        nb_classifier.learn(training_set)
        curr_avg = nb_classifier.test_model_effectiveness(test_set)
        total_effectiveness += curr_avg
    return total_effectiveness / k


def run(k: int = 5):
    """
    Main function of run_k_folded_cross_validation.py.

    param k: number of expected subsets
    type k: int
    """
    # get data
    dataset = get_data()
    # randomly shuffle given dataset
    shuffle(dataset)
    # run models
    avg_effectiveness = k_fold_validation(dataset, k)
    print("k_fold_validation average accuracy:", avg_effectiveness)
    return avg_effectiveness


if __name__ == "__main__":
    run()
