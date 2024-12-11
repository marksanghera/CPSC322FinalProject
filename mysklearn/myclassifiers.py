from mysklearn.myutils import tdidt, get_attribute_domains
# TODO: copy your myclassifiers.py solution from PA4-6 here

##############################################
# Programmer: Leo Jia
# Class: CptS 322-01, Fall 2024
# Programming Assignment #4
# 10/18/24
# 
# Description: This program does pa4
##############################################

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
import numpy as np

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()

        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = self.regressor.predict(X_test)
        y_predicted = [self.discretizer(pred) for pred in predictions]
        return y_predicted

def compute_euclidean_distance(v1, v2):
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []

        for test_instance in X_test:
            instance_distances = []
            for i, train_instance in enumerate(self.X_train):
                dist = compute_euclidean_distance(test_instance, train_instance)
                instance_distances.append((dist, i))

            instance_distances.sort(key = lambda x: x[0])

            top_k = instance_distances[:self.n_neighbors]

            distances.append([d[0] for d in top_k])
            neighbor_indices.append([d[1] for d in top_k])

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        distances, neighbor_indices = self.kneighbors(X_test)

        for indices in neighbor_indices:
            neighbor_labels = [self.y_train[i] for i in indices]
            
            label_counts = {}
            for label in neighbor_labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

            most_common_label = max(label_counts, key=label_counts.get)
            y_predicted.append(most_common_label)

        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        label_count = {}
        for label in y_train:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        self.most_common_label = max(label_count, key=label_count.get)


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label] * len(X_test)

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.priors = {}
        self.posteriors = {}

        # Calculate priors
        total_count = len(y_train)
        for label in y_train:
            if label not in self.priors:
                self.priors[label] = 0
            self.priors[label] += 1
        for label in self.priors:
            self.priors[label] /= total_count

        # Calculate posteriors
        feature_count = len(X_train[0])
        unique_labels = list(set(y_train))
        for feature_idx in range(feature_count):
            feature_values = [row[feature_idx] for row in X_train]
            self.posteriors[feature_idx] = {label: {} for label in unique_labels}
            for label in unique_labels:
                filtered_values = [feature_values[i] for i in range(len(y_train)) if y_train[i] == label]
                total_filtered = len(filtered_values)
                value_counts = {}
                for value in filtered_values:
                    if value not in value_counts:
                        value_counts[value] = 0
                    value_counts[value] += 1
                for value, count in value_counts.items():
                    self.posteriors[feature_idx][label][value] = count / total_filtered

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for x in X_test:
            label_probs = {}
            for label in self.priors:
                label_prob = self.priors[label]
                for feature_idx, feature_value in enumerate(x):
                    if feature_value in self.posteriors[feature_idx][label]:
                        label_prob *= self.posteriors[feature_idx][label][feature_value]
                    else:
                        label_prob *= 0
                label_probs[label] = label_prob
            predicted_label = max(label_probs, key=label_probs.get)
            y_predicted.append(predicted_label)
        return y_predicted


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        self.header = [f"att{i}" for i in range(len(X_train[0]))]
        self.attribute_domain = get_attribute_domains(X_train)
        avaliable_attributes = self.header.copy()
        self.tree = tdidt(train, avaliable_attributes, self.header, self.attribute_domain)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            y_predicted.append(self.tdidt_predict(self.tree, instance))
        return y_predicted

    def tdidt_predict(self, tree, instance):
        """Makes a prediction for a single instance.
        
        Args:
            tree: The decision tree
            instance: The instance to classify
            
        Returns:
            The predicted class label (True or False)
        """
        # base case: we are at a leaf node and can return the class prediction
        info_type = tree[0]  # "Leaf" or "Attribute"
        if info_type == "Leaf":
            return tree[1]  # class label

        # if we are here, we are at an Attribute
        # we need to match the instance's value for this attribute
        att_index = self.header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            # do we have a match with instance for this attribute?
            if value_list[1] == instance[att_index]:
                prediction = self.tdidt_predict(value_list[2], instance)
                if prediction is not None:
                    return prediction
    
        # If we get here, we didn't find a matching value or got a None prediction
        # Make a random choice between True and False
        import random
        return random.choice([True, False])

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        rules = []
        stack = [(self.tree, [])]

        while stack:
            current_node, conditions = stack.pop()

            if current_node[0] == "Leaf":
                rule = "IF " + " AND ".join(conditions) + f" THEN {class_name} = {current_node[1]}"
                rules.append(rule)
                continue

            att_index = current_node[1]
            att_name = attribute_names[att_index] if attribute_names else f"att{att_index}"

            for i in range(2, len(current_node)):
                value_list = current_node[i]
                if value_list[0] == "Value":
                    value = value_list[1]
                    new_condition = f"{att_name} == {value}"
                    stack.append((value_list[2], conditions + [new_condition]))

        for rule in rules:
            print(rule)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
