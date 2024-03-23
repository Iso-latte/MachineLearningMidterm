Resources:
 - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
 - https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a
 - https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter
 - Class materials
 - Chatgpt (Help with understanding plotting)

<h1><b>Machine Learning Midterm:</b>

- Models used:
    - Decision Tree Classifier
    - Logistic Regression
    - Support Vector Machine

<h1><b>REPORT:</b>

- <h2><b>Model One: Logistic Regression</b></h2>

    - Hyperparameters:
        - max iterations changed to 1000
            - This was because I was getting errors because of how many iterations the algorithm was doing. This seemed to help let the algorithm complete fully
    - Mean test time, mean fit time, and mean score time
      - Mean Test Score: 0.9733333333333334
      - Mean Fit Time: 0.019926166534423827
      - Mean Score Time: 0.0023434638977050783
    - Accuracy
      - Accuracy: 0.98
    - Classification Report (Numbers are the label encoded values)
      - Setosa [0] Precision: 1.00
      - Setosa [0] Recall: 1.00
      - Setosa [0] F1-Score: 1.00

      - Versicolor [1] Precision: 1.00
      - Versicolor [1] Recall: 0.94
      - Versicolor [1] F1-Score: 0.97

      - Virginica [2] Precision: 0.92
      - Virginica [2] Recall: 1.00
      - Virginica [2] F1-Score: 0.96

- <h2><b>Model Two: Support Vector Machine</b></h2>

    - Hyperparameters:
      - kernal('poly','rbf')
      - 'C' : [1,10]
      - svm.SVC()
    - Best parameters:
      - C = 10
      - kernal = 'rbf'
    - Mean test time, mean fit time, and mean score time
      - Mean Test Score: 0.9800000000000001
      - Mean Fit Time: 0.10444850921630859
      - Mean Score Time: 0.001841592788696289
    - Accuracy
      - Accuracy: 0.98
    - Classification Report (Numbers are the label encoded values)
      - Setosa [0] Precision: 1.00
      - Setosa [0] Recall: 1.00
      - Setosa [0] F1-Score: 1.00

      - Versicolor [1] Precision: 1.00
      - Versicolor [1] Recall: 0.94
      - Versicolor [1] F1-Score: 0.97

      - Virginica [2] Precision: 0.92
      - Virginica [2] Recall: 1.00
      - Virginica [2] F1-Score: 0.96


- <h2><b>Model Three: Decision Tree Classifier</b></h2>

    - Hyperparameters:
      - None
    - Mean test time, mean fit time, and mean score time
      - Mean Test Score: 0.9600000000000002
      - Mean Fit Time: 0.0024391651153564454
      - Mean Score Time: 0.0018893718719482423
    - Accuracy
      - Accuracy: 0.98
    - Classification Report (Numbers are the label encoded values)
      - Setosa [0] Precision: 1.00
      - Setosa [0] Recall: 1.00
      - Setosa [0] F1-Score: 1.00

      - Versicolor [1] Precision: 1.00
      - Versicolor [1] Recall: 0.94
      - Versicolor [1] F1-Score: 0.97

      - Virginica [2] Precision: 0.92
      - Virginica [2] Recall: 1.00
      - Virginica [2] F1-Score: 0.96


- <h2><b>Overall Conclusion:</b></h2>
  
  - Evaulation of Classification Reports
    - All the three models got the same scores which makes me think that the dataset was too small to get an effective conclusion about which one was the best. However, when looking at the scores given by the cross vaidation we can see that the models, while all very accurate, had different mean test scores. I can also see within all reports that a Veriscolor was incorrectly labeled as a Virginica which makes me think that they may share very similar properties.

  - Question: Which is the best classifier and why?
    - The best classifer for this data set would be the Support Vector Machine. The reason why I believe this is because of the mean test score upon cross validation was better than the other two at a 98%. However, it should be noted that the mean fit time was slower than the other two algorithms. I think the second best would be the Logistic Regression model which had the second best mean test score. I think that depending on the aims of the model such as accuracy or size that the best model will change. However, for this set i believe that the Support Vector Machine is the best choice.

  - Question: What affect the classification performance?
    - I think the things that affect the classification's performance the most are the hyperparameters. The reason why i believe this is because throughout this project I had changed the hyperparameters for the models and found that these were the best to wield these results. For example when I change the make iterations from 100 to 1000 this allowed the model to fully work. I think that these are the biggest factors in the affect of the classification performance of a given model.
