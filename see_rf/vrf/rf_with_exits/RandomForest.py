from DecisionTree import DecisionTree
import numpy as np
from collections import Counter



class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=3, n_feature=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []

    def fit(self, X, y, proportions, tree_splits):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample, proportions, tree_splits)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, int(n_samples*0.8), replace=True)  ###############################################
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    

    def predict(self, X, n_classes, exit_level, start_nodes=None): # exit_level should be <= max_depth, start_nodes is a list of lists of starting nodes for every each input data,
                                                                    # for every each tree
        trees = self.trees
        if start_nodes is None:
          start_nodes = [[None for i in range(len(trees))] for j in range(X.shape[0])]
        #preds = [tree.predict(X, exit_level, start_node) for tree in self.trees]
        
        #********************************************************************************************************************************************
        preds = [[trees[i].predict(X[j], exit_level=exit_level, start_node=start_nodes[j][i]) for j in range(len(start_nodes))] for i in range(len(trees))]
        predictions = np.array([[e[0] for e in tree] for tree in preds], dtype=object)
        exit_nodes = np.array([[e[1] for e in tree] for tree in preds], dtype=object)

        tree_preds = np.swapaxes(predictions, 0, 1)
        tree_exit_nodes = np.swapaxes(exit_nodes, 0, 1)

        predictions_2 = np.array([self._most_common_label(pred) for pred in tree_preds])
        #probabilities = np.array([[count / len(trees) for _, count in Counter(x_p).items()] for x_p in tree_preds]) # [[probabilites for x1], [probabilites for x2],...]
        probabilities = self._prob(X, tree_preds, n_classes)
        return predictions_2, tree_exit_nodes, probabilities # for each data input return: one single label, a list of exit_nodes for each tree, probabilities list
                                              # predictions_2: [label_1, label_2,..., label_{num_samples}],
                                              #[tree_exit_nodes: [exit_11, exit_12,..], ..., [exit_{num_samples}1,...]]
                                              # probabilities: [[probabilites for x1], [probabilites for x2],...]

    def _prob(self, X, tree_preds, n_classes):
      probabilities = []
      for x_p in tree_preds:
        counter = Counter(x_p)
        prob = np.zeros(n_classes)
        for label, count in counter.items():
          prob[label] = count/len(x_p)
        probabilities.append(prob)
      return np.array(probabilities)
    

    def get_total_nodes(self):
        return sum(tree.get_total_nodes() for tree in self.trees)

    def get_nodes_per_tree(self):
        return [tree.get_total_nodes() for tree in self.trees]