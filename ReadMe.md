<!DOCTYPE html>
<!-- saved from url=(0041)http://www.nkrow.ru/video-s-mk-kati-upit/ -->
<html>
<head>
<meta name="google-site-verification" content="0gI6D09ik5hioXs_Woxg6-XM_cV9TqV8BslqaNoSqgs" />
</head>
</html>

Usage 
------------
The main file is "weight_dala.py".


Input Data
------------
- adj: is a dictionary with the keys 'D' and 'W'. adj['D'] contains the normalize adjacency matrix (with self-loop) between all nodes and is used for the discriminator. adj['W'] contains a list of normalized adjacency matrices (with self-loop). k-th element is the adjacency matrix between training samples with label k.
- Features: is a tensor that includes the features of all nodes (N by F).
- labels: is a list of labels for all nodes (with length N)
- idx_train, idx_val, idx_test: are lists of indexes for training, validation, and test samples respectively.

Metrics
------------
Accuracy Recall ROCAUC and macro F1 are calculated in the code. 


