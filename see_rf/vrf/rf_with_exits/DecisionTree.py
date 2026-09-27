import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None, prop_level=None, end_proportion=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.prop_level = prop_level # which feature proportion level this node belongs to: first or second or third (1,2, or 3...); every each node will have this attribut
        self.end_proportion = end_proportion  # indicates that the node is in the end of a features proportion level

    def is_leaf_node(self):
        return self.left is None and self.right is None

    def is_end_proportion(self):
      return self.end_proportion != None


class DecisionTree:
    def __init__(self, min_samples_split=3, max_depth=10000, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self, X, y, proportions, tree_splits): # proportions is a list of percentages, the last percentage should be 1; same for tree splits
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y, prop=proportions, i=0, prop_level=1, tree_splits=tree_splits)



    def _grow_tree(self, X, y, prop=[], depth=0, i=0, prop_level=1, tree_splits=[]): # i here represents which proportion level and tree split we are at
            n_samples, n_feats = X.shape
            n_labels = len(np.unique(y))

            # check the stopping criteria
            if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
                value = self._most_common_label(y)
                return Node(value=value, prop_level=prop_level, end_proportion=1)

            # if we are no longer in the first percentage level so we add features starting from the last previous feature index (example: in level one we used 20% of features,
            # in level two we are targetting a data with 80% of the total features, so in this level we add only 60% of features since the other 20% have been already used earlier)
            if i>0:
              list_feat = np.arange(max(1, int(n_feats * prop[i-1])),max(1, int(n_feats * prop[i]))+1) # the list of features to sample from
              feat_idxs = np.random.choice( list_feat , max(1, int(n_feats * (prop[i]-prop[i-1]) * 0.8)), replace=False) # prop is the list of feature proportions

            else: # we are in the first level
              feat_idxs = np.random.choice( max(1, int(n_feats * prop[i])), max(1, int(n_feats * prop[i] * 0.8)), replace=False) # prop is the list of feature proportions
            

            # find the best split
            best_feature, best_thresh = self._best_split(X, y, feat_idxs)

            value = self._most_common_label(y) # every each node has a value

            # create child nodes
            left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
            # Whenever a prospective split would send every row to one side, the tree just stops and turns the current node into a leaf
            if len(left_idxs) == 0 or len(right_idxs) == 0:
                # can’t split any further – make the current node a leaf
                value = self._most_common_label(y)
                return Node(value=value, prop_level=prop_level, end_proportion=1)

            if (depth == int(self.max_depth * tree_splits[i])): # we are at the end of the current tree split and we are about to enter the next tree split
              i = i+1   # we need to go to the other proportion/ other tree split while expanding the children
              prop_level += 1 # we need to go to the other proportion level while expanding the children
              end_proportion = 1 # the current node is at the end of a proportion level

              left = self._grow_tree(X[left_idxs, :], y[left_idxs], prop, depth+1,i, prop_level, tree_splits=tree_splits)
              right = self._grow_tree(X[right_idxs, :], y[right_idxs], prop, depth+1,i, prop_level, tree_splits=tree_splits)
              return Node(best_feature, best_thresh, left, right, value=value, prop_level=prop_level-1, end_proportion = end_proportion)

            left = self._grow_tree(X[left_idxs, :], y[left_idxs], prop, depth+1,i, prop_level, tree_splits=tree_splits)
            right = self._grow_tree(X[right_idxs, :], y[right_idxs], prop, depth+1,i, prop_level, tree_splits=tree_splits)
            return Node(best_feature, best_thresh, left, right, value=value, prop_level=prop_level, end_proportion = None)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, x, exit_level, start_node=None):  # x is one single input data
        if start_node == None:
          start_node = self.root
        #return np.array([self._traverse_tree(x, exit_level, start_node) for x in X]) # array of tuples [(value_1, exit_node_1), (value_2, exit_node_2)]
        return self._traverse_tree(x, exit_level, start_node) #  (value, exit_node)

    def _traverse_tree(self, x, exit_level, node): # exit_level will be 1 or 2 or 3 ...
        if node.is_leaf_node() or (node.is_end_proportion() and node.prop_level == exit_level): # we stop when the node is a leaf or when get to the end of the target proportion
            exit_node = node
            return node.value, exit_node

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, exit_level, node.left)
        return self._traverse_tree(x, exit_level, node.right)
    
    def count_nodes(self, node):
        if node is None:
            return 0
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)
    
    def get_total_nodes(self):
        return self.count_nodes(self.root)