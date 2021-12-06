from math import pi, sqrt, exp


class NaiveBayesClassifier:
    """
    Represents single naive Bayes calssifier instance.
    """
    def __init__(self):
        """
        Constructor for class NaiveBayesClassifier.
        """
        self._data_by_classes = dict()
        self._numeric_characteristics = dict()
        self._training_set_size = 0

    def get_data_by_classes(self):
        """
        Getter for self._data_by_classes attribute.
        """
        return self._data_by_classes

    def get_numeric_characteristics(self):
        """
        Getter for self._numeric_characterisitcs attribute.
        """
        return self._numeric_characteristics

    def separate_by_classes(self, dataset: list):
        """
        Method takes the dataset and separates it by the classes of each
        sample. Class is defined as the last param in each of the data row of
        the given dataset.

        param dataset: represents dataset given to fit the classifier - list of
            list (each list represents one data sample)
        type dataset: list
        """
        data_by_classes = dict()
        for data_row in dataset:
            data_class = data_row[-1]
            if data_class not in data_by_classes:
                data_by_classes[data_class] = []
            data_by_classes[data_class].append(data_row[:-1])
        self._data_by_classes = data_by_classes

    def get_avg(self, data: tuple):
        """
        Method caluculates the average value of the given data.

        param data: data given for calculation
        type data: list
        """
        return sum(data) / float(len(data))

    def get_stdev(self, data):
        """
        Method caluculates the standard deviation (stdev) value of the given
        data.

        param data: data given for calculation
        type data: list
        """
        return sqrt(sum((value - self.get_avg(data)) ** 2 for value in data)
                    / float(len(data)))

    def calculate_numeric_characteristics(self):
        """
        Method calculates the numeric characteristics for each parametert of
        each class of the data recognised in previous steps of the learnig.
        """
        for class_key in self.get_data_by_classes():
            class_data = self.get_data_by_classes()[class_key]
            divided_into_attr_vect = [
                (
                    self.get_avg(column),
                    self.get_stdev(column),
                    len(column)
                )
                for column in zip(*class_data)
            ]
            if divided_into_attr_vect[1] == 0:
                print(divided_into_attr_vect[2])
                raise Exception
            self._numeric_characteristics[class_key] = divided_into_attr_vect

    def set_training_set_size(self):
        """
        Method calculates and sets the self._training_set_size attribute value
        as a size of the dataset given to fit the model.
        """
        total_set_size = sum([
            self.get_numeric_characteristics()[class_key][0][-1]
            for class_key in self.get_numeric_characteristics()
        ])
        self._training_set_size = total_set_size

    def get_training_set_size(self):
        """
        Getter for self._training_set_size attribute.
        """
        return self._training_set_size

    def learn(self, dataset: list):
        """
        Main method of the learning procedure of a NaiveBayesClassifier
        instance. Responsible for managing the process of learning.

        param dataset: represents dataset given to fit the classifier - list of
            list (each list represents one data sample)
        type classifier: list
        """
        # separate by classes
        self.separate_by_classes(dataset)
        # get numeric characteristics by classes
        self.calculate_numeric_characteristics()
        # set training_set_size attribute in order to ease latter calculations
        self.set_training_set_size()

    def calculate_probability(self, avg: float, stdev: float, x: float):
        """
        Mehtod calculates the probability using normal distribution density
        function.

        param avg: average value
        type avg: float

        param stdev: standard deviation
        type stdev: float

        param x: variable for which the method is suposed to find probability
        type x: float
        """
        return (1 / (sqrt(2 * pi)) * exp(-(x - avg) ** 2 / (2 * stdev ** 2)))

    def get_probabilites(self, data_sample: list):
        """
        Method calculates probabilities of belonging to each class for a given
        data sample.

        param data_sample: represents the data of uknown class - calculated
            probabilities will determin its class
        type data_sample: list
        """
        classes_probabilities = dict()
        for class_key in self.get_numeric_characteristics():
            characteristics = self.get_numeric_characteristics()[class_key]
            classes_probabilities[class_key] = (
                characteristics[0][-1] / float(self.get_training_set_size())
            )
            param_num = 0
            for param_stats in characteristics:
                avg, stdev, _ = param_stats
                # checking if stdev is different from 0
                # cif not, then there is no point in trying to use Bayesian...
                # because stdev equal to 0 causes ZeroDivisionError in
                # calculate_probability() method
                if stdev != 0:
                    curr_param_probability = self.calculate_probability(
                        avg, stdev, data_sample[param_num]
                    )
                    classes_probabilities[class_key] *= curr_param_probability
                param_num += 1
        return classes_probabilities

    def predict_class(self, data_sample):
        """
        Method responsible for predicting the data class of an object
        represented by the given data sample Method calls get_probabilities()
        method to get probabilities and then based on that data determines the
        class of the object.

        param data_sample: represents the data of uknown class - calculated
            probabilities will determin its class
        type data_sample: list
        """
        predicted_class = None
        predictions = self.get_probabilites(data_sample)
        max_probability = max([
            predictions[class_key]
            for class_key in predictions
        ])

        for class_key in predictions:
            if predictions[class_key] == max_probability:
                predicted_class = class_key
                break

        return predicted_class

    def test_model_effectiveness(self, test_dataset: list):
        """
        Method responisble for testing the accuracy of the model.
        Uses last parameter of each list in test_dataset to check predicted
        data class. Calculates average accuracy in predicting the class of
        data that has not been seen by the model during training session.

        param test_dataset: represents given test dataset - list of list (each
            list represents one data sample)
        type test_dataset: list
        """

        correct_predictions = 0
        for data_sample in test_dataset:
            predicted_class = self.predict_class(data_sample[:-1])
            if predicted_class == data_sample[-1]:
                correct_predictions += 1
        return correct_predictions / len(test_dataset)
