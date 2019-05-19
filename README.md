creditCardApproval
====================

In this project, we will build an automatic credit card approval predictor using machine learning techniques
Then create a simple API from a machine learning model in Python using Flask microFramework exposting the API to be used by other applications.

Structure of the application
--------------

Part 1. Building Machine Learning Model
    
    First, we will start off by loading and viewing the dataset.
    We will see that the dataset has a mixture of both numerical and non-numerical features, that it contains values from different ranges, plus that it contains a number of missing entries.
    We will have to preprocess the dataset to ensure the machine learning model we choose can make good predictions.
    After our data is in good shape, we will do some exploratory data analysis to build our intuitions.
    Finally, we will build a machine learning model that can predict if an individual's application for a credit card will be accepted.
    Using sklearn’s joblib, serialize the model(pickling)

Part 2. Converting machine learning model into APIs using Flask framework

  
    Load the already persisted model into memory when the application starts(depickling),
    Create an API endpoint that takes input variables, transforms them into the appropriate format, and returns predictions.
    
    
Structure of the Dataset
--------------
    Dataset is stored in csv file which each line containing a record of following fields

    data.frame':   689 obs. of  16 variables:
    *   $ Male          : num  1 1 0 0 0 0 1 0 0 0 ...
    *   $ Age           : chr  "58.67" "24.50" "27.83" "20.17" ...
    *   $ Debt          : num  4.46 0.5 1.54 5.62 4 ...
    *   $ Married       : chr  "u" "u" "u" "u" ...
    *   $ BankCustomer  : chr  "g" "g" "g" "g" ...
    *   $ EducationLevel: chr  "q" "q" "w" "w" ...
    *   $ Ethnicity     : chr  "h" "h" "v" "v" ...
    *   $ YearsEmployed : num  3.04 1.5 3.75 1.71 2.5 ...
    *   $ PriorDefault  : num  1 1 1 1 1 1 1 1 1 0 ...
    *   $ Employed      : num  1 0 1 0 0 0 0 0 0 0 ...
    *   $ CreditScore   : num  6 0 5 0 0 0 0 0 0 0 ...
    *   $ DriversLicense: chr  "f" "f" "t" "f" ...
    *   $ Citizen       : chr  "g" "g" "g" "s" ...
    *   $ ZipCode       : chr  "00043" "00280" "00100" "00120" ...
    *   $ Income        : num  560 824 3 0 0 ...
    *   $ Approved      : chr  "+" "+" "+" "+" ...
 
 Structure of TestData
--------------
    Test Data can be sent to API using JSON object via Postman client
    
    Fields "DriversLicense" and "ZipCode" are ignored while sending the request as these are ignored while building the ML model 

```
[
{"Gender": "b", "Age": "38.2", "Debt": 0.0, "Married": "u", "BankCustomer": "g", "EducationLevel": "w", "Ethnicity": "v", "YearsEmployed": 1.25, "PriorDefault": "t", "Employed": "t", "CreditScore": 1,  "Citizen": "g",  "Income":   0,"ApprovalStatus":"+"}
]
```


Execution and Testing
--------------

1) Execute flaskApp.py using Python3.6

2) Use Postman to send request to `http://127.0.0.1:9999/predict`
   *Note: Sample test data is present in 'datasets/testData/req.json'*
 