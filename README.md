# Welcome to RelAI!

**RelAI** is a Python library designed to calculate pointwise reliability for Machine Learning predictions.

## What is Machine Learning pointwise reliability?

Pointwise reliability measures how much trust we can place in the prediction for an individual instance. Imagine you've deployed a supervised machine learning model for a specific task, and you're using it to make predictions on new data. How confident can you be in these new predictions? Your model may have performed exceptionally well on the test dataset, leading you to believe that new predictions are likely to be accurate. However, what if the characteristics of the new cases differ from those in the training set, either due to changes over time or variations in the data space? What if the model doesn't generalize well to certain regions of the feature space, and your new case falls into one of those areas?

These scenarios can be understood through two principles that help us gauge the reliability of a prediction:

* **The Density Principle**: This principle assesses whether a sample is similar to the training set.
* **The Local Fit Principle**: This principle evaluates whether the model performed well on cases similar to the new case being classified.

RelAI incorporates these two principles, allowing you to analyze the reliability of predictions for each individual case when using your machine learning model.
