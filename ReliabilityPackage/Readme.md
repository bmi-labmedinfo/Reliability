# Reliability_Package

This repository includes the codes for the implementation of the Reliability package as implemented by Nicora et al. in [1]. 

## AIM

Provide a method for the assessment of the reliability of machine learning predictions on new unseen samples. 

## Background

Reliability is one of the key points to achieve trustworthy AI systems [2]. Like Saria et al. did [3], we use the term reliability to indicate the degree of trust of the prediction of a ML model on a single instance, and we build a method that relies on two fundamental principles: the density and the local fit principles. The density principle checks if the new test case is close to the training distribution, while the local fit principle checks if the classifier was accurate on the training samples closest to the new test case.

## Methods

The method implemented in this package computes the reliability of a new instance by evaluating the density and the local fit principles: a new instance is considered reliable if it is reliable according to both these principles, but they can also be applied separately.

### Density Principle

The density principle is implemented with the use of an Autoencoder: it is exploited to learn how to reproduce the training samples, so that samples coming from the same distribution of the training set are characterized by a low reconstruction error, while samples far from the training distribution (out-of-distribution samples) are characterized by a high reconstruction error. To assess the "density reliability" of a new unseen instance, the mean squared error (MSE) between the instance itself and its projection produced by the autoencoder is evaluated with respect to a threshold: if MSE <= threshold, then the prediction on such new instance can be considered "density reliable", while if MSE > threshold, it is considered "density unreliable". 

### Local Fit Principle

The local fit principle is implemented by training a classifier (i.e. an MLP) on a dataset of synthetic points generated ad-hoc to characterize the local performance of the classifier in the feature space; each synthetic point is associated with the performance value (accuracy for classification problems, mean squared error for regression problems) of its k closest training samples, and then labelled with respect to a performance threshold. 
In case of a classification problem, each synthetic point is labelled as "local fit reliable" if the accuracy value of its k nearest training samples is equal or higher than a certain accuracy threshold, "local fit unreliable" otherwise. 
In case of a regression problem, each synthetic point is labelled as "local fit reliable" if the Mean Squared Error of its k nearest training samples is equal or lower than a certain MSE threshold, "local fit unreliable" otherwise.
Finally, a classifier is trained on these so-labelled synthetic points, so that it learns how to classify new samples in terms of local fit reliability.

## License 

The Reliability_Package is released under the Creative Commons Attribution-NonCommercial 4.0 International License

## Contacts

For any question or information, please contact us at lorenzo.peracchio01@universitadipavia.it

## Documentation

Please find the Documentation of this package at https://rel-doc.readthedocs.io/en/latest/index.html

## Installation

1. Make sure you have the latest version of pip installed  
~~~python  
pip install --upgrade pip  
~~~
2. Install the ReliabilityPackage through pip  
~~~python  
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ReliabilityPackage 
~~~

## Usage

### Classification Problem

Here's a simple example of usage of the ReliabilityPackage for a typical classification problem, using the breast_cancer dataset of sklearn.  
1. import the needed functions from the package
~~~python 
from ReliabilityPackage.ReliabilityFunctions import *
~~~
2. import all the other necessary packages and functions
~~~python 
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.offline as pyo
~~~
3. load the breast cancer dataset and split it in a training, a validation, and a test set
~~~python 
X, y = datasets.load_breast_cancer(return_X_y=True)

X_seventy, X_test, y_seventy, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_seventy, y_seventy, test_size=0.30, random_state=42)
~~~
4. Train a classifier on the training set
~~~python 
clf = RandomForestClassifier(random_state=42, min_samples_leaf=10, n_estimators=100)
clf.fit(X_train, y_train)
~~~
5. Create and train an autoencoder for the implementation of the Density Principle  
(Please note that if the layer_sizes are not specified, the default autoencoder is built as follows: [dim_input, dim_input + 4, dim_input + 8, dim_input + 16, dim_input + 32]; if needed, specify a more suitable architecture)
~~~python 
ae = create_and_train_autoencoder(X_train, X_val, batchsize=80, epochs=1000)
~~~
6. Generate the dataset of the synthetic points and their associated values of accuracy
~~~python 
syn_pts, acc_syn_pts = generate_synthetic_points(problem_type = 'classification', predict_func=clf.predict, X_train=X_train, y_train=y_train, method='GN', k=5)
~~~
7. Define a Mean Squared Error threshold and an Accuracy threshold  
(the mse_threshold_plot can be generated to see how the performances change based on percentiles of the MSE of the validation set)
~~~python 
fig_mse_thresh = mse_threshold_plot(ae, X_val, y_val, clf.predict, metric = 'balanced_accuracy')
fig_mse_thresh.show()

mse_thresh = perc_mse_threshold(ae, X_val, perc=95)
acc_thresh = 0.90
~~~
8. Generate an instance of the ReliabilityDetector class for classification problems
~~~python 
RD = create_reliability_detector('classification', ae, syn_pts, acc_syn_pts, mse_thresh=mse_thresh, perf_thresh=acc_thresh, proxy_model="MLP")
~~~
9. It is now possible to compute the Reliability of the test_set
~~~python 
test_reliability= compute_dataset_reliability(RD, X_test, mode='total')
reliable_test = X_test[np.where(reliability_test == 1)]
unreliable_test = X_test[np.where(reliability_test == 0)]
~~~

### Regression Problem

Here's a simple example of usage of the ReliabilityPackage for a typical regression problem generated through the make_regression function of sklearn.  
1. import the needed functions from the package
~~~python 
from ReliabilityPackage.ReliabilityFunctions import *
~~~
2. import all the other necessary packages and functions
~~~python 
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
~~~
3. Generate a random regression dataset and split it in a training, a validation, and a test set
~~~python 
X, y = make_regression(n_samples=1000, n_features=20, noise=1, random_state=42)

X_seventy, X_test, y_seventy, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_seventy, y_seventy, test_size=0.30, random_state=42)
~~~
4. Train a linear regressor on the training set
~~~python 
reg = LinearRegression().fit(X_train, y_train)
~~~
5. Create and train an autoencoder for the implementation of the Density Principle  
(Please note that if the layer_sizes are not specified, the default autoencoder is built as follows: [dim_input, dim_input + 4, dim_input + 8, dim_input + 16, dim_input + 32]; if needed, specify a more suitable architecture)
~~~python 
ae = create_and_train_autoencoder(X_train, X_val, batchsize=80, epochs=1000)
~~~
6. Generate the dataset of the synthetic points and their associated values of Mean Squared Error
~~~python 
syn_pts, mse_syn_pts = generate_synthetic_points(problem_type = 'regression', predict_func=reg.predict, X_train=X_train, y_train=y_train, method='GN', k=5)
~~~
7. Define a Mean Squared Error threshold for the Density Principle and a performance threshold for the Local Fit Principle (MSE as the performance metric for the Local Fit Principle)
~~~python 
mse_thresh = perc_mse_threshold(ae, X_val, perc=95)
performance_thresh = 0.8
~~~
8. Generate an instance of the ReliabilityDetector class for regression problems
~~~python 
RD = create_reliability_detector('regression', ae, syn_pts, mse_syn_pts, mse_thresh=mse_thresh, perf_thresh=performance_thresh, proxy_model="MLP")
~~~
9. It is now possible to compute the Reliability of the test_set
~~~python 
test_reliability= compute_dataset_reliability(RD, X_test, mode='total')
reliable_test = X_test[np.where(reliability_test == 1)]
unreliable_test = X_test[np.where(reliability_test == 0)]
~~~

## Release History

0.0.1 (27-06-2023): initial release  
0.0.2 (27-06-2023): fixed bugs  
0.0.3 (28-06-2023): updated Readme  
0.0.4 (28-06-2023): refactoring (fix function name)  
0.0.5 (28-06-2023): adding function of average MSE and the validation loss plot in the training of the AE  
0.0.6 (28-06-2023): refactoring (update package dependencies)  
0.0.7 (28-06-2023): refactoring (update package dependencies)  
0.0.8 (29-06-2023): LICENSE update  
0.0.9 (29-06-2023): name errors fix  
0.0.10 (29-06-2023): update Readme.md  
0.0.11 (10-07-2023): update Readme.md  
0.0.12 (06-09-2023): updated functions' names  
0.0.13 (06-09-2023): fixed bugs  
0.0.14 (11-09-2023): fixed doc  
0.0.15 (11-09-2023): fixed doc  
0.0.16 (19-09-2023): fixed optimization parameters  
0.0.17 (10-11-2023): package updated for regression problems  
0.0.18 (10-11-2023): update Readme.md  
0.0.19 (10-11-2023): update Readme.md

## References
[1] Nicora G, Rios M, Abu-Hanna A, Bellazzi R. Evaluating pointwise reliability of machine learning prediction. Journal of Biomedical Informatics 2022;127:103996. https://doi.org/10.1016/j.jbi.2022.103996.  
[2]	Assessment List for Trustworthy Artificial Intelligence (ALTAI) for self-assessment | Shaping Europe’s digital future 2020. https://digital-strategy.ec.europa.eu/en/library/assessment-list-trustworthy-artificial-intelligence-altai-self-assessment.  
[3]	Saria S, Subbaswamy A. Tutorial: Safe and Reliable Machine Learning. ArXiv 2019; abs/1904.07204. https://doi.org/10.48550/arXiv.1904.07204.
