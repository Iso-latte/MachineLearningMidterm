# Machine Learning Midterm
## Link to Google Colab: https://colab.research.google.com/drive/1CiTe8qk4dF1GtFBJYtcgsdEzYeNo4CL9#scrollTo=C_Ddx2UQRDur&uniqifier=1

## Models Used:
- Decision Tree Classifier
- Logistic Regression
- Support Vector Machine

## Info
- The code is in ML.py
- This tests three different ML models
- The code does cross validation for each model
- It also graphs the confusion matrix of each model

## Requirments
- SKlearn
- Numpy
- Pandas
- Matplotlib

# REPORT

## Model One: Logistic Regression

### Hyperparameters:
- max iterations changed to 1000
  - This was because I was getting errors due to the number of iterations the algorithm was performing. Increasing it seemed to help the algorithm complete fully.
- Mean test time, mean fit time, and mean score time:
  - Mean Test Score: 0.9733333333333334
  - Mean Fit Time: 0.019926166534423827
  - Mean Score Time: 0.0023434638977050783
- Accuracy:
  - Accuracy: 0.98
- Classification Report (Numbers are the label encoded values):
  - Setosa [0] Precision: 1.00
  - Setosa [0] Recall: 1.00
  - Setosa [0] F1-Score: 1.00
  - Versicolor [1] Precision: 1.00
  - Versicolor [1] Recall: 0.94
  - Versicolor [1] F1-Score: 0.97
  - Virginica [2] Precision: 0.92
  - Virginica [2] Recall: 1.00
  - Virginica [2] F1-Score: 0.96

## Model Two: Support Vector Machine

### Hyperparameters:
- Kernel: 'poly','rbf'
- 'C': [1,10]
- svm.SVC()
- Best parameters:
  - C = 10
  - Kernel = 'rbf'
- Mean test time, mean fit time, and mean score time:
  - Mean Test Score: 0.9800000000000001
  - Mean Fit Time: 0.10444850921630859
  - Mean Score Time: 0.001841592788696289
- Accuracy:
  - Accuracy: 0.98
- Classification Report (Numbers are the label encoded values):
  - Setosa [0] Precision: 1.00
  - Setosa [0] Recall: 1.00
  - Setosa [0] F1-Score: 1.00
  - Versicolor [1] Precision: 1.00
  - Versicolor [1] Recall: 0.94
  - Versicolor [1] F1-Score: 0.97
  - Virginica [2] Precision: 0.92
  - Virginica [2] Recall: 1.00
  - Virginica [2] F1-Score: 0.96

## Model Three: Decision Tree Classifier

### Hyperparameters:
- None
- Mean test time, mean fit time, and mean score time:
  - Mean Test Score: 0.9600000000000002
  - Mean Fit Time: 0.0024391651153564454
  - Mean Score Time: 0.0018893718719482423
- Accuracy:
  - Accuracy: 0.98
- Classification Report (Numbers are the label encoded values):
  - Setosa [0] Precision: 1.00
  - Setosa [0] Recall: 1.00
  - Setosa [0] F1-Score: 1.00
  - Versicolor [1] Precision: 1.00
  - Versicolor [1] Recall: 0.94
  - Versicolor [1] F1-Score: 0.97
  - Virginica [2] Precision: 0.92
  - Virginica [2] Recall: 1.00
  - Virginica [2] F1-Score: 0.96

## Overall Conclusion:

### Evaluation of Classification Reports:
- All three models achieved the same scores, suggesting that the dataset may be too small to draw conclusive differences. However, cross-validation scores differed slightly, indicating varying performance.
- The mislabeling of Versicolor as Virginica across all reports suggests they may share similar properties.

### Which is the best classifier and why?
- The Support Vector Machine appears to be the best classifier for this dataset due to its higher mean test score of 98%. However, its fit time is slower than the other models. Logistic Regression follows closely with the second-best mean test score. The best model choice may vary based on priorities such as accuracy or computational efficiency.

### What affects the classification performance?
- Hyperparameters seem to be the most significant factor affecting classification performance. Adjusting hyperparameters such as the number of iterations greatly impacted model performance. These parameters play a crucial role in determining the effectiveness of a given model.
