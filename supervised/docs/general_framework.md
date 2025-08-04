## General Framework of Graph Neural Networks

Graph neural networks iteratively update node representations by combining the representations of their neighbors and their own representations. Each layer has two important functions:

**AGGREGATE:** To aggregate information from neighbors of each node.

**COMBINE:** To update the node representations by combining the aggregated information from neighbors with the current node representations.
