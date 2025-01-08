import pandas as pd
import numpy as np
from scipy.sparse import find


def get_tree_description(clf, features):

    # Create a list to store the node information
    nodes_dicts = []

    # Tree parameters
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    for i in range(n_nodes):
        if is_leaves[i]:

            nodes_dicts.append({
                'node': i,
                'node_type': 'leaf',
                'values': values[i]
            })

        else:

            nodes_dicts.append({
                'node': i,
                'node_type': 'split',
                'values': values[i],
                'feature': features[feature[i]],  #X.iloc[:, feature[i]].name,
                'threshold_lte': threshold[i],
                'branch_true': children_left[i],
                'branch_false': children_right[i]
            })

    return pd.DataFrame(nodes_dicts)


def print_full_tree_description(tree, features):
    def gini(a, b):
        '''
        Calculate node impurity
        '''
        return 1 - (a/(a+b))**2 - (b/(a+b))**2

    nodes_dicts = get_tree_description(tree, features)
    split_nodes = nodes_dicts.loc[nodes_dicts['node_type']=='split'].astype({'branch_true': 'int', 'branch_false': 'int'})

    # Create a DataFrame for the leaf nodes
    leaf_nodes = nodes_dicts.loc[nodes_dicts['node_type']=='leaf'].drop(columns=['feature', 'threshold_lte', 'branch_true', 'branch_false'])

    # Split the valies in class 0 and class 1
    leaf_nodes[['class_0', 'class_1']] = leaf_nodes['values'].apply(lambda x: x[0]).apply(pd.Series)
    leaf_nodes = leaf_nodes.drop(columns='values')

    # Calculate gini for each leaf
    leaf_nodes['gini'] = leaf_nodes.apply(lambda row: gini(row['class_0'], row['class_1']), axis=1)

    # Determine the dominant leaf class
    leaf_nodes['at_risk'] = 1*(leaf_nodes['class_1'] > leaf_nodes['class_0'])

    return split_nodes, leaf_nodes


def get_decision_boundaries(model, scaler, features):
    # Get the full description of the tree (information about all nodes)
    split_nodes, leaf_nodes = print_full_tree_description(model, features)

    # Create a DataFrame with the aggregated (count) values of threshold for each feature
    feature_boundaries = pd.DataFrame(split_nodes['feature'].value_counts()).reset_index(drop=False)

    # Create empty columns for metrics
    feature_boundaries['min_th'] = 0
    feature_boundaries['max_th'] = 0
    feature_boundaries['nb_unique_values'] = 0

    for i, row in feature_boundaries.iterrows():
        mask = (split_nodes['feature']==row['feature'])
        feature_boundaries.loc[i, 'min_th'] = split_nodes.loc[mask]['threshold_lte'].min()
        feature_boundaries.loc[i, 'max_th'] = split_nodes.loc[mask]['threshold_lte'].max()
        feature_boundaries.loc[i, 'nb_unique_values'] = split_nodes.loc[mask]['threshold_lte'].nunique()

    # Create a DataFrame with the features in the same order as used in the model
    feature_names = pd.DataFrame({'feature': model.feature_names_in_})

    # Merge on feature name
    feature_boundaries['feature'] = feature_boundaries['feature'].astype(str)  # Convert to string
    feature_names['feature'] = feature_names['feature'].astype(str)  # Convert to string
    feature_boundaries = pd.merge(feature_names, feature_boundaries, on='feature', how='left')

    # Create a DataFrame of min and max threshold value to use in scaler.inverse_transform
    minmax = feature_boundaries[['feature', 'min_th', 'max_th']]
    minmax.index = feature_boundaries['feature']
    minmax = minmax.drop(columns='feature').T.reset_index(drop=True)

    # Replace the scaled values with non-scaled values in the DataFrame
    feature_boundaries['min_th'] = scaler.inverse_transform(minmax)[0]
    feature_boundaries['max_th'] = scaler.inverse_transform(minmax)[1]

    # Create a DataFrame with each node and the branching logic
    node_switch_leaf = leaf_nodes[['node']]
    node_switch_leaf['branch_true'] = node_switch_leaf['node']
    node_switch_leaf['branch_false'] = node_switch_leaf['node']
    node_switch_leaf = node_switch_leaf.rename(columns={
        'node': 'start',
        'branch_true': 'end_1',
        'branch_false': 'end_2'
    })

    node_switch = split_nodes[['node', 'branch_true', 'branch_false']]
    node_switch = node_switch.rename(columns={
        'node': 'start',
        'branch_true': 'end_1',
        'branch_false': 'end_2'
    })

    node_switch = pd.concat([node_switch, node_switch_leaf])

    # Set column to True if the node is a leaf (end node)
    node_switch['leaf'] = node_switch['start'] == node_switch['end_1']

    # Create a list to store all the possible paths
    paths = []

    # Start from each end node to get up the tree
    for end_node in node_switch.loc[node_switch['leaf']==True].index:
        # Initialise the path
        path = []

        # Start at the end node
        node = end_node

        # Navigate through the nodes until the root node
        while node != 0:
            # Add the node to the path
            path.append(node)

            # Find the previous node (i.e. what node linked to the current node)
            conditions = (node_switch['leaf']==False) & ((node_switch['end_1']==node) | (node_switch['end_2']==node))

            # Set the previous node as the current node
            node = int(node_switch.loc[conditions]['start'])

        # Add Node 0 to the path
        path.append(0)

        # Flip the path (from root to leaf)
        path = path[::-1]

        # Add path to list of paths
        paths.append(path)

    boundaries = []
    for select_path in paths:

        # Save leaf node in its own DataFrame
        last_node = leaf_nodes.loc[leaf_nodes['node']==select_path[-1]]

        # Get risk label from leaf node
        risk_label = int(last_node['at_risk'])
        impurity = float(last_node['gini'])

        # Save split nodes in their own DataFrame
        split_path = split_nodes.loc[split_nodes['node'].isin(select_path[:-1])].reset_index(drop=True)

        # Check whether the Threshold was met or not (i.e. value <= threshold)
        threshold_test = []
        for i, row in split_path.iterrows():

            if i < (len(split_path)-1):
                threshold_test.append(row['branch_true'] == split_path.loc[i+1, 'node'])
            else:
                threshold_test.append(row['branch_true'] == int(last_node['node']))

        # Add threshold test to path DataFrame
        split_path['threshold_test'] = threshold_test

        # Save only relevant columns
        split_path = split_path[['feature', 'threshold_lte', 'threshold_test']]

        boundaries.append({'at_risk': risk_label,
                           'gini': impurity,
                           'feature_thresholds': split_path.to_dict('records'),
                           'nodes_on_path': select_path})

    # Transform results into a DataFrame
    boundaries_df = pd.DataFrame(boundaries)

    # Separate the DataFrame based on risk classification
    lower_risk_df = boundaries_df.loc[boundaries_df['at_risk']==0].reset_index(drop=True)
    higher_risk_df = boundaries_df.loc[boundaries_df['at_risk']==1].reset_index(drop=True)

    # Add columns to count the number of branches from the node
    # leading to a risk classification of 1 or 0
    split_nodes['branches_risk_left'] = 0
    split_nodes['branches_risk_right'] = 0

    # Check whether the classification risk incrases or decreases
    # on the left and right subnodes
    for i, row in split_nodes.iterrows():
        start_node = row['node']
        next_node_left = row['branch_true']
        next_node_right = row['branch_false']

        # Calculate the faction of class-1 on the left branch
        left_path = set([start_node, next_node_left])
        left_class_0 = lower_risk_df['nodes_on_path'].apply(lambda x: left_path.issubset(x)).sum()
        left_class_1 = higher_risk_df['nodes_on_path'].apply(lambda x: left_path.issubset(x)).sum()
        split_nodes.loc[i, 'branches_risk_left'] = left_class_1 / (left_class_0+left_class_1)

        # Calculate the faction of class-1 on the right branch
        right_path = set([start_node, next_node_right])
        right_class_0 = lower_risk_df['nodes_on_path'].apply(lambda x: right_path.issubset(x)).sum()
        right_class_1 = higher_risk_df['nodes_on_path'].apply(lambda x: right_path.issubset(x)).sum()
        split_nodes.loc[i, 'branches_risk_right']  = right_class_1 / (right_class_0+right_class_1)

    # Set the inverse flag if the risk decreases when the feature value increases
    split_nodes['inverse'] = split_nodes['branches_risk_left'] > split_nodes['branches_risk_right']

    # Set the no-change flaf if the risk doesn't change to the left and right subnodes
    split_nodes['no_change'] = split_nodes['branches_risk_left'] == split_nodes['branches_risk_right']

    # Use a negative sign if the inverse flag is True
    split_nodes['inverse'] = split_nodes['inverse'].apply(lambda x: -1 if x else 1)

    return boundaries_df, split_nodes


def get_decision_path(model, x_scaled):
    # Get the decision path for each sample
    decision_paths = model.decision_path(x_scaled)

    # Extract the paths as non-zero indices
    sample_id, node_id, _ = find(decision_paths)

    return [list(node_id[sample_id == i]) for i in range(len(x_scaled))]