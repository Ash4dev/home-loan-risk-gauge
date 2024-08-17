# Home Credit Loan Workflow

## Introduction

Understanding the credit lending process is crucial for financial institutions to mitigate risk and make informed lending decisions. Each step in the workflow plays a vital role in evaluating loan applications and determining creditworthiness. This document outlines the importance of each step in the context of home credit loans and details the specific methods used to analyze and predict default probabilities.

## Background

In the home credit lending industry, predicting the likelihood of loan defaults is essential for maintaining financial stability and managing risk. The workflow involves several key steps:

1. **Home Pricing Analysis**: Determining the market value of properties using historical data to assess the collateral value of loans.
2. **Customer Risk Assessment**: Analyzing customer data to evaluate their creditworthiness and predict the probability of default.

## Data Sources

1. **Home Pricing Dataset**: This dataset contains information about various home features and their market prices. It is sourced from Kaggle's [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

2. **Customer Data**: This dataset includes customer information relevant to credit risk assessment. It is available from Kaggle's [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data).

## Analysis and Techniques

### 1. Home Pricing Analysis

To determine property values, regression techniques are applied to the home pricing dataset. The aim is to predict the selling price of homes based on various features.

### 2. Customer Risk Assessment

To evaluate customer default probabilities, classification techniques are employed on the customer data. Various models are used to predict the likelihood of loan default.

### Techniques Used

Whenever we make any important decision we first discuss it with friends, family or an expert. Nowadays we check the reviews on social media or check a YouTube video. Considering other people's opinion just make final decision more informed and make sure to avoid any kind of surprises as we are combining multiple opinions about the same thing together.

Ensemble modeling in machine learning operates on the same principle, where we combine the predictions from multiple models to generate the final model which provide better overall performance. Ensemble modeling helps to generalize the learning based on training data, so that it will be able to do predictions accurately on unknown data.

Modeling is one of the most important step in machine learning pipeline. The main motivation behind ensemble learning is to correctly combine weak models to get a more accurate and robust model with bias-variance trade off. For example Random Forest algorithm is ensemble of Decision Tree and since it combine multiple decision tree models it always perform better than single decision tree model.

Depending on how we combine the base models, ensemble learning can be classified in three different types Bagging, Boosting and Stacking.
- Bagging: The working principle is to build several base models independently and then to average them for final predictions.
- Boosting: Boosting models are built sequentially and tries to reduce the bias on final predictions.
- Stacking: The predictions of each individual model are stacked together and used as input to a final estimator to compute the prediction.

Ensemble learning approach makes the model more robust and helps to achieve the better performance.

#### Bagging

![Bagging](https://cdn.analyticsvidhya.com/wp-content/uploads/2023/08/image-7.png)

- In bagging we build independent estimators on different samples of the original data set and average or vote across all the predictions.
- Bagging is a short form of *Bootstrap Aggregating. It is an ensemble learning approach used to improve the stability and accuracy of machine learning algorithms.
- Since multiple model predictions are averaged together to form the final predictions, Bagging reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method.
- Bagging is a special case of the model averaging approach, in case of regression problem we take mean of the output and in case of classification we take the majority vote.
- Bagging is more helpfull if we have over fitting (high variance) base models.
- We can also build independent estimators of same type on each subset. These independent estimators also enable us to parallelly process and increase the speed.
- Most popular bagging estimator is 'Bagging Tress' also knows as 'Random Forest'

##### Bootstrap

![Bootstrap](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/Bagging.png)

- It is a resampling technique, where large numbers of smaller samples of the same size are repeatedly drawn, with replacement, from a single original sample.
- So this technique will enable us to produce as many subsample as we required from the original training data.
- So the defination is simple to understand, but "replacement" word may be confusing sometimes. Here 'replacement' word signifies that the same obervation may repeat more than once in a given sample, and hence this technique is also known as sampleing with replacement
- As you can see in above image we have training data with observations from X1 to X10. In first bootstrap training sample X6, X10 and X2 are repeated where as in second training sample X3, X4, X7 and X9 are repeated.
- Bootstrap sampling helps us to generate random sample from given training data for each model in order to genralise the final estimation.

So in case of Bagging we create multiple number of bootstrap samples from given data to train our base models. Each sample will contain training and test data sets which are different from each other and remember that training sample may contain duplicate observations.

#### Boosting

![Boosting](https://media.geeksforgeeks.org/wp-content/uploads/20210707140911/Boosting.png)

- In case of boosting, machine learning models are used one after the other and the predictions made by first layer models are used as input to next layer models. The last layer of models will use the predictions from all previous layers to get the final predictions.
- So boosting enables each subsequent model to boost the performance of the previous one by overcomming or reducing the error of the previous model.
- Unlike bagging, in case of boosting the base learners are trained in sequence on a weighted version of the data. Boosting is more helpful if we have biased base models.
- Boosting can be used to solve regression and classification problems.

#### Stacking

![Stacking](https://media.geeksforgeeks.org/wp-content/uploads/20200713234827/mlxtend.PNG)

- Stacking is a way to ensemble multiple classifications or regression model. There are many ways to ensemble models, the widely known techniques are Bagging or Boosting. Bagging allows multiple similar models with high variance which are averaged to decrease variance. Boosting builds multiple incremental models to decrease the bias, while keeping variance small.

- Stacking is a different paradigm. The point of stacking is to explore a space of different models for the same problem. The idea is that you can attack a learning problem with different types of models which are capable to learn some part of the problem, but not the whole space of the problem. So, you can build multiple different learners and you use them to build an intermediate prediction, one prediction for each learned model. Then you add a new model which learns from the intermediate predictions for the same target.

This final model is said to be stacked on the top of the others, hence the name. Thus, you might improve your overall performance, and often you end up with a model which is better than any individual intermediate model.

### When to use Ensemble Learning?
Since Ensemble learning results in better accuracy, high consistency and also helps to avoid bias variance tradeoff should'nt we use it everywhere? The short answer is it depends on the problem in hand. If our model with available training data is not performing well and showing the signs of overfitting/unterfitting and additinal compute power is not an issue then going for Ensemble Learning is best option. However one shouldnt skip the first steps of improving the input data and trying different hyperparmeters before going for ensemple approach.

## Models used

For the various tasks
1. regression (house price prediction): ensemble learners mostly used such as XGBoost, LightGBM, stacked Elastic Net & Kernel Ridge regression
2. classification (loan default): ensemble learners outperformed baseline logistic regression, random forest models

## Getting Started

### Prerequisites

Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

### Installation

1. **Clone the Repository**

   Open your terminal and run the following command to clone the repository:

   ```bash
   git clone https://github.com/yourusername/home-credit-loan-workflow.git
   cd home-credit-loan-workflow

2.Create a virtual environment and install the required packages using pip:
	```
	python -m venv venv
	source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
	pip install -r requirements.txt
	```

# Running the Analysis

1. Home Pricing Analysis
	```
	python home_pricing_analysis.py
	```
2. Probability of Default, Loss given default, Expected losses
	```
	python risk_assessment.py
	```
	
# Conclusion
This workflow provides a comprehensive approach to assessing home credit loans by analyzing home pricing and customer risk. The methods applied offer practical insights and experience in handling real-world data for credit risk evaluation.

Feel free to explore the code and adapt the techniques as needed for your specific use cases. For further details and updates, refer to the repository's documentation and commit history.
