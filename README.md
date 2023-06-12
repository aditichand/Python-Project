# Python-Project

## Loan Defaulter Prediction Models

_Disclaimer: This project was done in a team of 6 as the final deliverable for a Master's Level Machine Learning course. The dataset and case description were chosen by the team.<br>_

# Table of Contents
* [Problem Statement](https://github.com/aditichand/Python-Project/blob/main/README.md#problem-statement)
* [Introduction](https://github.com/aditichand/Python-Project/blob/main/README.md#introduction)
* [Dataset Description](https://github.com/aditichand/Python-Project/blob/main/README.md#dataset-description)
* [Data Wrangling](https://github.com/aditichand/Python-Project/blob/main/README.md#data-wrangling)
* [Running Initial Models](https://github.com/aditichand/Python-Project/blob/main/README.md#running-initial-models) 
* [Hyperparameter Tuning](https://github.com/aditichand/Python-Project/blob/main/README.md#hyperparameter-tuning)
* [Conclusion](https://github.com/aditichand/Python-Project/blob/main/README.md#conclusion)

 
### Problem Statement 
Assemble a machine learning model using various algorithms to compare and predict a customer's loan eligibility based on various parameters

### Introduction
This case is about a bank which has a growing customer base. Most of these customers are depositors. The number of borrowers is quite small, and the bank is interested in expanding their base to bring more loan business and, in the process, earn more through the interest on loans. The management wants to explore potential loan customers. The department wants to build a model that will help them to identify those potential customers who have a higher probability of getting approved for and repaying the loan so that it will increase their success ratio while keeping risks to a minimum.

### Motivation
Loan defaulting is a significant financial risk for the banking industry as it damages the interests of lenders and breaks the social trust. Researchers have made extensive effort into developing efficient machine learning techniques to help regulators carry out an accurate loan approval process in real-time. Through our effort in this project, we hope to create and identify a model that can best help discern potential defaulters.


### Dataset Description**
The data set used includes 1,25,000 observations with eighteen key variables and among them we have identified four different measurement categories (Loan Status, Term, Home Ownership, Purpose) and converted them into categorial variables. Our independent variable is **Loan Status**, as we want to see what factors are likely to determine whether a loan will be paid off or not, and this pattern of factors can be used for the evaluation of loan approvals in the future.<br>

We divided the data set in a split of Test and Train in 20% and 80% respectively<br> 
Total Dataset Entries: 1,25,000 Rows x 18 Columns<br>
Test Dataset Entries: 25,000 Rows x 16 Columns<br>
Train Dataset Entries: 1,00,000 Rows x 16 Columns<br>
Data Source: https://www.kaggle.com/code/manarkandeel/bank-loans-logistic-regression-knn/data<br>

### Data Cleaning
<br>
#### _Transforming the incorrect data_
In our data set we have observed that 4540 rows have credit score greater than 800. This is not possible since maximum credit score is 800.
The 4540 rows have credit score greater than 800 were mistakenly multiplied by 10 and hence the error had occurred on our dataset, we therefore transformed the following rows of data by dividing the faulty credit score it by 10 and thereby correcting the data of the rows.
 
#### _Imputing the Missing Values_
Next, we identified the rows with missing values. Missing values can be replaced by the minimum, maximum or average value of that Attribute. Zero can also be used to replace missing values. Any replenishment value can also be specified as a replacement of missing values. We identified 19111 rows with missing “Credit Score”, “Years in current job” and “Annual Income”. Since, it is one of our KPI’s and is essential for our analysis. A reliable machine learning modeling demands for careful handling of missing data and hence we try to replace the values with logic.<br>
<br>
Since the dataset consists of great outliers for “Annual Income”, we have chosen replacement by median to imputing the missing values for “Annual Income” and the “Credit Score.”<br>
“Years in current job” the data is observed to be skewed, and hence it is good to consider using Mode values for imputing missing data with Mode values for the data in “Years in current job”.

#### _Preprocessing and Encoding Categorical Data_
Machine learning models also require all input and output variables to be numeric and if the data contains categorical data, we must encode it to numbers before we can fit and evaluate the model. Encoding categorical data is a process of converting categorical data into integer format so that the data with converted categorical values can be provided to the models to give and improve the predictions. Our Dataset contains four different measurement categories (Loan Status, Term, Home Ownership, Purpose) which are among the KPI’s for our model, hence we transformed non- numerical labels or categorical variables (as long as they are hashable and comparable) to numerical labels using LabelEncode (Fit label encoder and return encoded labels)

#### _Dropping Unnecessary Columns_
The Loan ID and Customer ID do not add any interesting information, so they will be neglected in the problem. Then we have identified the number of missing values in our data set and the rows corresponding to small number of missing values (Bankruptcies, Tax Liens, Maximum) are dropped.

### **Running initial models**
After performing the EDA and checking the relationship of the independent variables with the variable we want to predict, we decided to build 5 models and analyze the performance of these to select the one that best aligns with our objective. On running the models without any hyperparameter tuning, we got the following results:
The models overfit the training data and gave poor performances in the testing data, so we performed hyperparameter tuning on our models as follows:

<img width="814" alt="image" src="https://github.com/aditichand/Python-Project/assets/61296787/6d74b2db-e36f-480f-8e9d-a1665033dee4">

### **Hyperparameter Tuning**
1. Logistic Regression : No hyperparameter tuning<br>
2. KNN : n_neighbors = 3, from elbow plot<br>
3. Decision Tree<br>
Parameter ranges: max_features {2:12}, max_depth {2:10} Final parameters: max_features=6, max_depth=4<br>
4. Random Forest<br>
Parameter ranges: max_features {2:12}, max_depth {2:10}<br>
Final Parameters: n_estimators=50, max_features=5, max_depth=6)<br>
5. XG Boost<br>
Objective = "binary:logistic"<br>
Final Parameters: n_estimators=1000, random state = 42)<br>

On running the models with the new parameters, we got the following results:
<img width="790" alt="image" src="https://github.com/aditichand/Python-Project/assets/61296787/035a22ea-bc63-46d8-a9a2-6a4dbd72e76a">

For banks, the rate of True Positives is important because there will be a huge cost associated with categorizing a customer as someone who has paid off their loan when they have in fact, not done so. Recall score calculates how many of the Actual Positives our model captures through labeling it as a True Positive. Due to this business necessity, we decided to use the Recall score as the primary metric to select the best model when there is a high cost associated with False Negative.
F1 Score might be a better measure to use if we need to seek a balance between Precision and Recall and when there is an uneven class distribution (large number of Actual Negatives). So, for our purpose, we evaluated the models based on the Recall score. Decision Tree had the highest Recall rate out of all the models at 99.5%. In addition, the test accuracy for Decision Tree is also the highest among all models, solidifying it as the best model for our objective.

<img width="475" alt="image" src="https://github.com/aditichand/Python-Project/assets/61296787/6396e630-1c19-463b-84fb-b35ea6ab3891">

But there are a few problems associated with choosing decision tree as our model:<br>
**1. Sensitive to Change**: A small change in training data can result in a completely different tree. The overall accuracy might still be high, but the specific decision splits will be totally different. So, we are not sure how the performance will be change if there are changes in the data.<br>
**2. Cannot Deal with Class Imbalance**: Classification decision trees, as is in our case, tend to favor predicting the dominant class in datasets with class imbalance.<br>

### **Conclusion**
On running all the models, we saw that despite the hyperparameter tuning, the testing accuracy of the models remained largely unchanged, lying between 60-70%. This could signify that the features in the dataset are not appropriate for prediction of our independent variable, i.e., they do not have a significant relationship with the variable, or we might just need more data to increase the accuracy of our models. Obtaining more data will also help to solve the problem of class imbalance in our current dataset.
