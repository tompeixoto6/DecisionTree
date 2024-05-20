import pandas as pd
import math
import sys
from collections import Counter

class Node:
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = []
        self.count = 0

class DecisionTree:
    def __init__(self, X, feature_names, labels):
        self.X = X
        self.feature_names = feature_names
        self.labels = labels
        self.labelCategories = list(set(labels))
        self.labelCategoriesCount = [labels.count(x) for x in self.labelCategories]
        self.root = None
        self.entropy = self._get_entropy(list(range(len(self.labels))))
        self.bins = {}  # Store bin boundaries for numeric features

    def _get_entropy(self, x_ids):
        labels = [self.labels[i] for i in x_ids]
        label_count = [labels.count(x) for x in self.labelCategories]
        entropy = sum([-count / len(x_ids) * math.log(count / len(x_ids), 2) if count else 0 for count in label_count])
        return entropy

    def _get_information_gain(self, x_ids, feature_id):
        info_gain = self._get_entropy(x_ids)
        x_features = [self.X[x][feature_id] for x in x_ids]
        feature_vals = list(set(x_features))
        feature_v_count = [x_features.count(x) for x in feature_vals]
        feature_v_id = [[x_ids[i] for i, x in enumerate(x_features) if x == y] for y in feature_vals]

        info_gain_feature = sum([v_counts / len(x_ids) * self._get_entropy(v_ids) for v_counts, v_ids in zip(feature_v_count, feature_v_id)])
        info_gain -= info_gain_feature
        return info_gain

    def _get_feature_max_information_gain(self, x_ids, feature_ids):
        features_entropy = [self._get_information_gain(x_ids, feature_id) for feature_id in feature_ids]
        max_id = feature_ids[features_entropy.index(max(features_entropy))]
        return self.feature_names[max_id], max_id

    def predict(self, instance):
        def classify(node, instance):
            if not node.childs:
                return node.value
            attribute_index = self.feature_names.index(node.value)
            attribute_value = instance[attribute_index]
            for child in node.childs:
                if self._match_value(attribute_value, child.value):
                    return classify(child.next, instance)
            return max(self.labelCategories, key=self.labels.count)
        
        return classify(self.root, instance)

    def _match_value(self, attribute_value, node_value):
        if isinstance(node_value, tuple):
            lower, upper = node_value
            return lower <= attribute_value < upper
        return attribute_value == node_value
    
    def id3(self, x_ids, feature_ids, node=None):
        if not node:
            node = Node()
        
        labels_in_features = [self.labels[x] for x in x_ids]

        if len(set(labels_in_features)) == 1:
            node.value = labels_in_features[0]
            node.count = len(labels_in_features)
            return node

        if len(feature_ids) == 0:
            node.value = max(set(labels_in_features), key=labels_in_features.count)
            node.count = len(labels_in_features)
            return node

        best_feature_name, best_feature_id = self._get_feature_max_information_gain(x_ids, feature_ids)
        node.value = best_feature_name
        node.childs = []

        feature_values = list(set([self.X[x][best_feature_id] for x in x_ids]))

        for value in feature_values:
            child = Node()
            child.value = value
            node.childs.append(child)
            child_x_ids = [x for x in x_ids if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                child.next = Node()
                child.next.value = max(set(labels_in_features), key=labels_in_features.count)
                child.next.count = len(labels_in_features)
            else:
                new_feature_ids = feature_ids[:]
                if best_feature_id in new_feature_ids:
                    new_feature_ids.remove(best_feature_id)
                child.next = self.id3(child_x_ids, new_feature_ids, child.next)

        return node

    def preprocess_data(self):
        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                if pd.isna(self.X[i][j]):
                    if isinstance(self.X[i][j], str):
                        self.X[i][j] = "None"
                    else:
                        self.X[i][j] = self._get_most_common_value(j)

    def _get_most_common_value(self, feature_id):
        feature_values = [self.X[i][feature_id] for i in range(len(self.X)) if not pd.isna(self.X[i][feature_id])]
        if not feature_values:
            return None
        return max(set(feature_values), key=feature_values.count)

    def discretize_numeric_features(self, bins=3):
        for i in range(len(self.X[0])):
            if isinstance(self.X[0][i], (int, float)):
                column_values = [row[i] for row in self.X]
                min_val, max_val = min(column_values), max(column_values)
                bin_width = (max_val - min_val) / bins
                bin_edges = [min_val + j * bin_width for j in range(bins)] + [max_val]
                bin_edges = [round(edge, 2) for edge in bin_edges]  # Round bin edges
                self.bins[i] = [(bin_edges[j], bin_edges[j+1]) for j in range(bins)]
                for j in range(len(self.X)):
                    for k in range(bins):
                        if bin_edges[k] <= self.X[j][i] < bin_edges[k + 1]:
                            self.X[j][i] = (bin_edges[k], bin_edges[k + 1])
                            break

def read_csv(file_path):
    df = pd.read_csv(file_path)
    df = df.drop('ID', axis=1)
    feature_names = list(df.columns[:-1])
    X = df[feature_names].values.tolist()
    labels = df[df.columns[-1]].values.tolist()

    return X, feature_names, labels

def read_test_csv(file_path):
    df = pd.read_csv(file_path)
    return df.values.tolist()

def build_tree_dict(node):
    if not node.childs:
        return node.value
    tree_dict = {}
    for child in node.childs:
        tree_dict[child.value] = build_tree_dict(child.next)
    return {node.value: tree_dict}

def print_tree(tree, depth=0):
    if isinstance(tree, dict):
        for key, value in tree.items():
            print("  " * depth + f"<{key}>")
            print_tree(value, depth + 1)
    else:
        print("  " * depth + str(tree))

def classify(tree, instance, feature_names):
    if not isinstance(tree, dict):
        return tree
    for attribute, subtree in tree.items():
        attribute_index = feature_names.index(attribute)
        attribute_value = instance[attribute_index]
        if attribute_value in subtree:
            return classify(subtree[attribute_value], instance, feature_names)
    return None

def get_value_distribution(X, labels):
    distributions = {}
    for feature_index in range(len(X[0])):
        feature_values = [row[feature_index] for row in X]
        value_counts = Counter(feature_values)
        distributions[feature_index] = dict(value_counts)
    label_distribution = Counter(labels)
    return distributions, label_distribution

def main():
    if len(sys.argv) < 2:
        print("Please provide a dataset file as a command-line argument.")
        return

    dataset_file = sys.argv[1]

    try:
        X, feature_names, labels = read_csv(dataset_file)
    except FileNotFoundError:
        print(f"Error: The file '{dataset_file}' was not found.")
        return

    print(f"Feature Names: {feature_names}")
    print(f"Labels: {list(set(labels))}")

    tree = DecisionTree(X, feature_names, labels)
    tree.preprocess_data()
    tree.discretize_numeric_features()
    tree.root = tree.id3(list(range(len(tree.X))), list(range(len(tree.feature_names))))

    tree_dict = build_tree_dict(tree.root)
    distributions, label_distribution = get_value_distribution(X, labels)

    print(f"Value Distributions: {distributions}")
    print(f"Label Distribution: {label_distribution}")

    print("Tree Structure:")
    print_tree(tree_dict)

    if len(sys.argv) > 2:
        test_file = sys.argv[2]
        try:
            test_data = read_test_csv(test_file)
            for instance in test_data:
                classification = tree.predict(instance)
                instance_dict = dict(zip(feature_names, instance))
                print(f"Instance: {instance_dict} -> Classification: {classification}")
        except FileNotFoundError:
            print(f"Error: The test file '{test_file}' was not found.")
            return

if __name__ == "__main__":
    main()

