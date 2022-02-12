import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
import utils

#setup the randoms tate
RANDOM_STATE = 19920604


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


#input: Y_pred,Y_true
#output: accuracy, auc, precision, recall, f1-score
def classification_metrics(Y_pred, Y_true):
	#NOTE: It is important to provide the output in the same order
	return accuracy_score(Y_true, Y_pred)

#input: Name of classifier, predicted labels, actual labels
#output: print ACC, AUC, Prec, Recall and F1-Score of the Classifier
def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print("______________________________________________")
	print("")

def main():
    #load training and testing data
    X_train, Y_train = utils.get_data_from_svmlight("features_svmlight.train")
    X_test, Y_test = utils.get_data_from_svmlight("features_svmlight.validate")

    #change and select params here
    param_kfold = 10 #parameter k for kfold CV
    param_n_split = 5 #number of splits for shufflesplit CV
    param_kernel = 'linear' #type of kernel for SVM classifier
    param_c = 0.2 #regularization parameter for SVM classifier
    
    #setting figsize and axes
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    
    #SVC
    title = "Learning Curves (SVM, " + param_kernel + " kernel)"
    cv = ShuffleSplit(n_splits=param_n_split, test_size=0.2, random_state=RANDOM_STATE)    
    estimator = SVC(kernel=param_kernel,C=param_c)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:, 0], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    estimator.fit(X_train,Y_train)
    display_metrics("SVC linear",estimator.predict(X_test),Y_test)
    
    param_kernel = 'rbf'
    title = "Learning Curves (SVM, " + param_kernel + " kernel)"
    cv = ShuffleSplit(n_splits=param_n_split, test_size=0.2, random_state=RANDOM_STATE)    
    estimator = SVC(kernel=param_kernel,C=param_c)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:, 1], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    estimator.fit(X_train,Y_train)
    display_metrics("SVC rbf",estimator.predict(X_test),Y_test)
    
    param_kernel = 'poly'
    title = "Learning Curves (SVM, " + param_kernel + " kernel)"
    cv = ShuffleSplit(n_splits=param_n_split, test_size=0.2, random_state=RANDOM_STATE)    
    estimator = SVC(kernel=param_kernel,C=param_c)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:, 2], ylim=(0.4, 1.01), cv=cv, n_jobs=4
    )
    estimator.fit(X_train,Y_train)
    display_metrics("SVC poly",estimator.predict(X_test),Y_test)
    
    plt.show()
	

if __name__ == "__main__":
	main()
	
