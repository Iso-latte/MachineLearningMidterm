import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


#  loading the test.csv from the MNIST data set
data = pd.read_csv('dataset/iris.csv')


#  Used to change the labels to numerical values
label_encoder = LabelEncoder()
data['variety'] = label_encoder.fit_transform(data['variety'])


#  Taking all the data and putting in var X
X = data[data.columns]
#  Drop target variable (Blair helped me with this)
X = data.drop('variety', axis=1)
#  Taking the label of the data collected
Y = data["variety"]


#  splitting testing and training data (testing is 30%)
x_train, x_test, y_train, y_test = train_test_split (X,Y,test_size=0.3,random_state=0)


#  The commented out lines are to see how the label encoder coded the labels
#print(data['variety'])
#print(label_encoder.inverse_transform(data['variety']))
classes = ['Setosa', 'Versicolor', 'Virginica']


#  Define the confusion matrix plot function
#  This allows us to use this function to plot all confusion matrices without having to repeat code
def plot_confusion_matrix(confusion_matrix, title):
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

'''Model One: Logistic Regression'''

model_one = LogisticRegression(max_iter=1000)

#  Fitting the data to Model one
model_one.fit(x_train, y_train)
y_pred_one = model_one.predict(x_test)


#  Cross validation
cv_results_one = cross_validate(model_one, X, Y, cv=5)
print('\n'+'*'*20+'\tLogistic Regression\t'+'*'*20+'\n\n')
print('Mean test score: ' + str(np.mean(cv_results_one['test_score'])))
print('Mean fit time: ' + str(np.mean(cv_results_one['fit_time'])))
print('Mean score time: ' + str(np.mean(cv_results_one['score_time'])))


#  Accuracy, and classification report
print('\nAccuracy of Logistic Regression on test set: {:.2f}'.format(model_one.score(x_test, y_test)))
class_report_one = classification_report(y_test, y_pred_one)
print('\nClassification Report:\n', class_report_one)


#Calculating and printing the confusion matrix
confusion_matrix_one = confusion_matrix(y_test, y_pred_one)
print('Confusion Matrix:\n')
print(confusion_matrix_one)
print('\n')

#  Plot for confusion matrix
plt.figure(figsize=(5, 3))
plot_confusion_matrix(confusion_matrix_one,'Confusion Matrix for Logistic Regression')
plt.show()

'''Model Two: Support Vector Machine'''

#Setting up the svm model's parameters and using Grid search
parameters = {'kernel':('poly', 'rbf'), 'C':[1,10]}
svc = svm.SVC()
model_two = GridSearchCV(svc, parameters)


#  Fitting data to Model Two
model_two.fit(x_train, y_train)
y_pred_two = model_two.predict(x_test)


#  Calculating the best score
print('\n'+'*'*20+'\tSupport Vector Machine\t'+'*'*20+'\n\n')
best_params = model_two.best_params_
print('Best Parameters: '+ str(best_params))


#  Cross validation
cv_results_two = cross_validate(model_two, X, Y, cv=5)
print('Mean test score: ' + str(np.mean(cv_results_two['test_score'])))
print('Mean fit time: ' + str(np.mean(cv_results_two['fit_time'])))
print('Mean score time: ' + str(np.mean(cv_results_two['score_time'])))


#  Accuracy, classification report
print('\nAccuracy of Support Vector Machine on test set: {:.2f}'.format(model_two.score(x_test, y_test)))
class_report_two = classification_report(y_test, y_pred_two)
print('\nClassification Report:\n', class_report_two)

#  Calculating and printing the confusion matrix
confusion_matrix_two = confusion_matrix(y_test, y_pred_two)
print('Confusion Matrix:\n')
print(confusion_matrix_two)
print('\n')

#  Plot for confusion matrix
plt.figure(figsize=(5, 3))
plot_confusion_matrix(confusion_matrix_two,'Confusion Matrix for Support Vector Machine')
plt.show()

'''Model Three: Decision Tree Classifier'''
model_three = DecisionTreeClassifier()


#  Fitting data to Model Three
model_three.fit(x_train, y_train)
y_pred_three = model_three.predict(x_test)


#  Cross validation
cv_results_three = cross_validate(model_three, X, Y, cv=5)
print('\n'+'*'*20+'\tDecision Tree Classifier\t'+'*'*20+'\n\n')
print('Mean test score: ' + str(np.mean(cv_results_three['test_score'])))
print('Mean fit time: ' + str(np.mean(cv_results_three['fit_time'])))
print('Mean score time: ' + str(np.mean(cv_results_three['score_time'])))


#  Accuracy, classification report
print('\nAccuracy of Decision Tree classifier on test set: {:.2f}'.format(model_three.score(x_test, y_test)))

class_report_three = classification_report(y_test, y_pred_three)
print('\nClassification Report:\n', class_report_three)


#Calculating and printing the confusion matrix
confusion_matrix_three = confusion_matrix(y_test, y_pred_three)
print('Confusion Matrix:\n')
print(confusion_matrix_three)
print('\n')


#  Plot for confusion matrix
plt.figure(figsize=(5, 3))
plot_confusion_matrix(confusion_matrix_three,'Confusion Matrix for Decision Tree Classifier')
plt.show()