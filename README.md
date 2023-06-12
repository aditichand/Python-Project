# Python-Project
**Dataset Description**

The data set includes 1,25,000 observations with eighteen key variables and among them we have identified four different measurement categories (Loan Status, Term, Home Ownership, Purpose) and converted them into categorial variables. Our independent variable is Loan Status, as we want to see what factors are likely to determine whether a loan will be paid off or not, and this pattern of factors can be used for the evaluation of loan approvals in the future.
We divided the data set in a split of Test and Train in 20% and 80% respectively Total Dataset Entries: 1,25,000 Rows x 18 Columns
Test Dataset Entries: 25,000 Rows x 16 Columns
Train Dataset Entries: 1,00,000 Rows x 16 Columns

_Transforming the incorrect data:_

In our data set we have observed that 4540 rows have credit score greater than 800. This is not possible since maximum credit score is 800.
The 4540 rows have credit score greater than 800 were mistakenly multiplied by 10 and hence the error had occurred on our dataset, we therefore transformed the following rows of data by dividing the faulty credit score it by 10 and thereby correcting the data of the rows.
 
_Imputing the Missing Values:_

Next, we identified the rows with missing values. Missing values can be replaced by the minimum, maximum or average value of that Attribute. Zero can also be used to replace missing values. Any replenishment value can also be specified as a replacement of missing values. We identified 19111 rows with missing “Credit Score”, “Years in current job” and “Annual Income”. Since, it is one of our KPI’s and is essential for our analysis. A reliable machine learning modeling demands for careful handling of missing data and hence we try to replace the values with logic.
Since the dataset consists of great outliers for “Annual Income”, we have chosen replacement by median to imputing the missing values for “Annual Income” and the “Credit Score”
“Years in current job” the data is observed to be skewed, and hence it is good to consider using Mode values for imputing missing data with Mode values for the data in “Years in current job”.
_Preprocessing and Encoding Categorical Data:_

Machine learning models also require all input and output variables to be numeric and if the data contains categorical data, we must encode it to numbers before we can fit and evaluate the model. Encoding categorical data is a process of converting categorical data into integer format so that the data with converted categorical values can be provided to the models to give and improve the predictions. Our Dataset contains four different measurement categories (Loan Status, Term, Home Ownership, Purpose) which are among the KPI’s for our model, hence we transformed non- numerical labels or categorical variables (as long as they are hashable and comparable) to numerical labels using LabelEncode (Fit label encoder and return encoded labels)
_Dropping Unnecessary Columns:_

The Loan ID and Customer ID do not add any interesting information, so they will be neglected in the problem. Then we have identified the number of missing values in our data set and the rows corresponding to small number of missing values (Bankruptcies, Tax Liens, Maximum) are dropped.
After performing the EDA and checking the relationship of the independent variables with the variable we want to predict, we decided to build 5 models and analyze the performance of these to select the one that best aligns with our objective. On running the models without any hyperparameter tuning, we got the following results:
The models overfit the training data and gave poor performances in the testing data, so we performed hyperparameter tuning on our models as follows:

PM<img width="814" alt="image" src="https://github.com/aditichand/Python-Project/assets/61296787/6d74b2db-e36f-480f-8e9d-a1665033dee4">

_Hyperparameter Tuning_
1. Logistic Regression : No hyperparameter tuning
2. KNN : n_neighbors = 3, from elbow plot
3. Decision Tree
Parameter ranges: max_features {2:12}, max_depth {2:10} Final parameters: max_features=6, max_depth=4
4. Random Forest
Parameter ranges: max_features {2:12}, max_depth {2:10}
Final Parameters: n_estimators=50, max_features=5, max_depth=6)
5. XG Boost
Objective = "binary:logistic"
Final Parameters: n_estimators=1000, random state = 42)

On running the models with the new parameters, we got the following results:
