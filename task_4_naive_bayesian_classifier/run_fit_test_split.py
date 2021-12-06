from get_data import get_data
from naive_bayes_classifier import NaiveBayesClassifier
from random import shuffle


def split_dataset(dataset: list, n: float = 0.6):
    """
    Function splits dataset into 2 parts in given proportion n.

    param dataset: represents dataset given to fit the classifier - list of
            list (each list represents one data sample)
    type classifier: list

    param n: represents the proportion between fit and test datasets sizes
    type n: float
    """
    # split dataset
    idx = round(len(dataset) * n)
    return dataset[:idx], dataset[idx:]


def get_fit_test_effectiveness(
    dataset: list,
    n: float = 0.6,
    is_test_set_equal_fit_set: bool = False
):
    """
    Function creates, fits and tests the Naive Bayesian Classifier.

    param dataset: represents dataset given to fit the classifier - list of
        list (each list represents one data sample)
    type classifier: list

    param n: represents the proportion between fit and test datasets sizes
    type n: float

    param is_test_set_equal_fit_set: answers the question if test and fit sets
        are the same
    type is_test_set_equal_fit_set: bool
    """
    training_set, test_set = split_dataset(dataset, n)
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.learn(training_set)
    if not is_test_set_equal_fit_set:
        return nb_classifier.test_model_effectiveness(test_set)
    else:
        return nb_classifier.test_model_effectiveness(training_set)


def run(n: float = 0.6, is_test_set_equal_fit_set: bool = False):
    """
    Main function of run_fit_test_split.py.

    param n: represents the proportion between fit and test datasets sizes
    type n: float

    param is_test_set_equal_fit_set: answers the question if test and fit sets
        are the same
    type is_test_set_equal_fit_set: bool
    """
    # get data
    dataset = get_data()
    # randomly shuffle given dataset
    shuffle(dataset)
    # run models
    avg_effectiveness = get_fit_test_effectiveness(
        dataset, n, is_test_set_equal_fit_set
    )
    print("fit_test average accuracy:", avg_effectiveness)
    return avg_effectiveness


if __name__ == "__main__":
    run()
