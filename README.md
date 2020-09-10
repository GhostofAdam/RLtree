## Readme

#### Feature Histogram

Right images are the baseline splitting feature histogram, and the right are RL tree.

Index -2 represents the leaf node.

It is not true that the RL tree uses much more feature than baseline CART tree.

* breast_cancer

<figure class="half">
    <img src="./histogram/breast_cancer.png" figcaption="Baseline" width="350" >
    <img src="./histogram_RL/breast_cancer.png" figcaption="RL" width="350" >
</figure>

* german

<figure class="half">
    <img src="./histogram/german.png" title="Baseline" width="350" >
    <img src="./histogram_RL/german.png" title="RL" width="350" >
</figure>

* heart

<figure class="half">
    <img src="./histogram/heart.png" title="Baseline" width="350" >
    <img src="./histogram_RL/heart.png" title="RL" width="350" >
</figure>

* pima

<figure class="half">
    <img src="./histogram/pima.png" title="Baseline" width="350" >
    <img src="./histogram_RL/pima.png" title="RL" width="350" >
</figure>

#### A New Tree Policy Network

Inspired by Tree-LSTM.

Policy Network: $F$
Inputs: Node Embedding $x$, Parent's Topology Embedding: $h_p$
Outputs: Action $a$, Topology Embedding: $h_{true},h_{false}$
$$
h_{true},h_{false},a = F(x,\hat{h_p})
$$
