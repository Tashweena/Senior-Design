# Equalmodel

This repository contains code for Mean Multicalibration, Moment Multicalibration, and Alternating Gradient Descent. 

Moment Multicalibration: finding “regression functions which, given a data point, can make point predictions not just for the expectation of its label, but for higher moments of its label distribution as well—and those predictions match the true distribution quantities when averaged not just over the population as a whole, but also when averaged over an enormous number of finely defined subgroups” (Roth et. al. 2020, https://arxiv.org/abs/2008.08037). 

## Mean Multi calibration
* Calibration roughly means that the predictor should mean what it says. For example, if we see in a dataset that if it rains 10% of the time, in my predictor, it should have also rained 10% of the time. More formally, let the true label(y) be between 0 and 1. We can divide this range in m buckets so that we have [ 0, 1/m, …,]. A predictor is said to be mean-multicalibrated if for all subgroups that are defined, and for each bucket, if the true mean is close to i, the predicted mean is also close to i.

* The 2 key components of the algorithm are the auditor and the fixer (i.e projected gradient descent). The subgroups can be either manually pre-defined or selected by a learning oracle consistency auditor. 

### Learning Oracle
* Instead of enumerating all the subgroups, we can define an algorithm that will learn to select the subsets in our data that have high residuals(difference between prediction and true label)
 
* The input to the algorithm are the features(X), the true residuals. The goal is to learn a function such that, in every iteration, given another set of points the learning oracle learns a function and can predict the residuals for those points. The points that have predicted residuals above a certain threshold are passed to the Auditor.
The learning oracle has been implemented but testing is ongoing. We experimented with linear regression and stochastic gradient descent with logistic loss so far.


### Auditor
* The role of the auditor is to identify whether the predictions for a subgroup is alpha-mean consistent. If the difference between the predicted mean and the true mean of a subgroup is above a certain threshold, the auditor will identify a consistency violation. It will return -1 if the model is currently underestimating, +1 if it is overestimating the predictions, None otherwise. The auditor can tolerate some small degrees of deviation from the true mean prediction, defined by the hyperparameters alpha and delta.


### Fixer (i.e. Projected Gradient Descent)
* The goal of projected gradient descent is to fix the biased prediction through multiple iterations. It is used in two aspects: post-processing and prediction. 
For post-processing, in each iteration the auditor will first indicate if the prediction of a selected subgroup is over- or under-estimated. Next, we adjust (add or subtract) the prediction of that subgroup by α, and project the adjusted prediction within [0,1]. Then, we store the action for that specific subgroup. This process stops after convergence, that is after the auditor does not find any subgroups that need to be fixed. For each iteration, we use a fresh batch of data to avoid overfitting.

* For prediction, we use a new set of test data. First, we generate the initial prediction from the given machine learning model. We then use projected gradient descent from the same sequence of actions saved from post-processing to adjust the initial predictions of subgroups.

Here is how all the parts come together for mean multicalibration:
<img width="800" alt="Screen Shot 2021-04-05 at 9 22 07 PM" src="https://user-images.githubusercontent.com/66379483/113645593-fc982b80-963b-11eb-85fd-016e40deca79.png">


## Moment Multi calibration 
* The goal of moment multicalibration is to calibrate every defined or created subgroup to converge to the true variance of the subgroup, with an error of β (user-defined) allowed. We can divide this range of means, and moments in m buckets so that we have [ 0, 1/m, …,] for each of them. A predictor is said to be moment-multicalibrated if for all subgroups that are defined and for all points whose predicted mean lies in the i-th bucket, if the true moment is close to j, the predicted moment is also close to j. 

* So far, only the second moment has been implemented, but the algorithm can calibrate higher moments as well. 

* The 2 key components of the algorithm are the auditor and the fixer (i.e projected gradient descent). The subgroups can be either manually pre-defined or selected by a learning oracle consistency auditor. 
 
Here is how all the parts come together for moment multicalibration:
<img width="800" alt="Screen Shot 2021-04-05 at 9 24 10 PM" src="https://user-images.githubusercontent.com/66379483/113645683-35380500-963c-11eb-92c8-5f3ffbf0910e.png">

## Alternating Gradient Descent 
* The goal of alternating gradient descent is to ensure that every predefined subgroup or subgroup selected by a learning oracle consistency auditor is both mean and moment calibrated. After calibration, uncertainty estimates can be generated for each prediction made by the original model.

Here is how all the parts come together for alternating gradient descent: 
<img width="800" alt="Screen Shot 2021-04-05 at 9 25 24 PM" src="https://user-images.githubusercontent.com/66379483/113645777-61538600-963c-11eb-971c-0229aaf5d2bb.png">

## Dataset
* We used the CIFAR10 dataset which is an image dataset comprising 50K training images and 10K test images. There are 10 classes: airplane, automobile, bird, cat,  deer, dog, frog, horse, ship and truck. For our purposes, we split the 60K images into 30K images for training and 20K images for validation and we are predicting whether the object in the image can fly or not. We predefined subgroups as two binary categories: (1) in_water, out_water (2) is_animal, is_not_animal. 

* The use of the CIFAR10 was motivated by the need for very large datasets to avoid overfitting and the main goal was to test the statistical properties of the algorithms.

* We also used a linear dataset with noise x + ε = y, where ε ~ N(0, 1) for a classification task. If y is above or equal to 0.5, the class label is 1 and if the y is below 0.5, the class label is 0.

## Instructions to Reproduce Our Results
For demonstration of the Moment Multicalibration algorithm, please see our jupyter notebook.



