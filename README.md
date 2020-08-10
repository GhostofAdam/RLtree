## README

* RL Decision Tree framework

  Predict the splitting feature of each node, the splitting value is still the value of maximum information gain.

  Loss is based on REINFORCE, a Policy Gradient method

* Performance on Iris

  Traditional Decision Tree with 0.9 accuracy on test data.

  ![](https://github.com/GhostofAdam/RLtree/blob/master/tree_images/normal_iris.png)

  

  Reinforce Learning Tree with 0.9 on test data.

  ![](https://github.com/GhostofAdam/RLtree/blob/master/tree_images/iris_99.png)

  Training Process

  ![](https://github.com/GhostofAdam/RLtree/blob/master/tree_images/iris.gif)