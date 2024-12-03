import numpy as np # use numpy's random number generation

# TODO: your reusable general-purpose functions here

def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]


def tdidt(current_instances, available_attributes, header, attribute_domains):
    print("available attributes:", available_attributes)
    # basic approach (uses recursion!!):
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes)
    print("splitting on:", split_attribute)
    available_attributes.remove(split_attribute)  # can't split on this attribute again
    tree = ["Attribute", split_attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, header, attribute_domains)
    print("partitions:", partitions)
    # for each partition, repeat unless one of the following occurs (base case)
    for att_value in sorted(partitions.keys()):  # process in alphabetical order
        att_partition = partitions[att_value]
        value_subtree = ["Value", att_value]
        # CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            print("CASE 1")
            value_subtree.append(["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)])
        # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            print("CASE 2")
            value_subtree.append(["Leaf", majority_vote(att_partition), len(att_partition), len(current_instances)])
        # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            print("CASE 3")
            continue
        else:
            # none of the base cases were true, recurse
            subtree = tdidt(att_partition, available_attributes.copy(), header, attribute_domains)
            value_subtree.append(subtree)
        tree.append(value_subtree)
    return tree


def select_attribute(instances, attributes):
    def calculate_entropy(partition):
        total = len(partition)
        if total == 0:
            return 0
        label_counts = {}
        for instance in partition:
            label = instance[-1]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        entropy = 0
        for count in label_counts.values():
            probability = count / total
            entropy -= probability * (0 if probability == 0 else np.log2(probability))
        return entropy

    min_entropy = float('inf')
    best_attribute = None

    for attribute in attributes:
        attribute_index = int(attribute[3:])  # Extract attribute index from "att#"
        # Partition the instances based on the current attribute's values
        partitions = {}
        for instance in instances:
            att_value = instance[attribute_index]
            if att_value not in partitions:
                partitions[att_value] = []
            partitions[att_value].append(instance)
        # Calculate the weighted average entropy for the partitions
        total_instances = len(instances)
        weighted_entropy = 0
        for partition in partitions.values():
            partition_entropy = calculate_entropy(partition)
            weighted_entropy += (len(partition) / total_instances) * partition_entropy
        # Update the best attribute if this one has a lower entropy
        if weighted_entropy < min_entropy:
            min_entropy = weighted_entropy
            best_attribute = attribute

    return best_attribute

def partition_instances(instances, attribute, header, attribute_domains):
    # this group by attribute domain (not values of attribute in instances)
    # lets use dictionaries
    att_index = header.index(attribute)
    att_domain = attribute_domains[attribute]
    partitions = {}
    for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions


def all_same_class(instances):
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False
    # get here, then all same class labels
    return True

def get_attribute_domains(X_train):
        """Extracts the domains for each attribute from the training data.

        Args:
            X_train(list of list of obj): The list of training instances (samples).

        Returns:
            dict: A dictionary where keys are attribute indexes and values are sets of unique values.
        """
        attribute_domains = {}
        for att_index in range(len(X_train[0])):
            attribute_domains[f"att{att_index}"] = set(instance[att_index] for instance in X_train)
        return attribute_domains

def majority_vote(instances):
        label_counts = {}
        for instance in instances:
            label = instance[-1]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        return max(label_counts, key=label_counts.get)