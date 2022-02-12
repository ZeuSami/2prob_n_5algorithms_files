import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
import utils

#setup the randoms tate
RANDOM_STATE = 19920604


#input: X_train, Y_train, X_test, Y_test
#output: Decision Tree Classifier with the optimal alpha
def decisionTree_optimal_alpha(X_train, Y_train, X_test, Y_test):
	#train a decision tree classifier using X_train and Y_train. Use this to predict labels of X_train
    #find the optimal alpha value for pruning
    max_acc = 0.0
    optimal_ccp_alpha = 0
    # Picking alpha from data applicable alphas
    path = DecisionTreeClassifier().cost_complexity_pruning_path(X_train,Y_train)
    ccp_alphas = path.ccp_alphas    
    for ccp_alpha in ccp_alphas:
          clf = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=ccp_alpha)
          clf.fit(X_train,Y_train)
          acc = accuracy_score(Y_test, clf.predict(X_test))
          if max_acc < acc:
              clf_w_optimal_alpha = clf
              max_acc = acc
              optimal_ccp_alpha = ccp_alpha
    print("Optimal alpha used:" + str(optimal_ccp_alpha))
    return clf_w_optimal_alpha


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
	return accuracy_score(Y_true, Y_pred),roc_auc_score(Y_true, Y_pred),precision_score(Y_true, Y_pred),recall_score(Y_true, Y_pred),f1_score(Y_true, Y_pred)

#input: Name of classifier, predicted labels, actual labels
#output: print ACC, AUC, Prec, Recall and F1-Score of the Classifier
def display_metrics(classifierName,Y_pred,Y_true):
	print("______________________________________________")
	print(("Classifier: "+classifierName))
	acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: "+str(acc)))
	print(("AUC: "+str(auc_)))
	print(("Precision: "+str(precision)))
	print(("Recall: "+str(recall)))
	print(("F1-score: "+str(f1score)))
	print("______________________________________________")
	print("")

def main():
    #load training and testing data
    X_train, Y_train = utils.get_data_from_svmlight("features_svmlight.train")
    X_test, Y_test = utils.get_data_from_svmlight("features_svmlight.validate")

    #change and select params here
    param_kfold = 10 #parameter k for kfold CV
    param_alpha = 0.01 #complexity parameter alpha for decision tree
    
    #setting figsize and axes
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    #DTC
    title = "Learning Curves Decision Tree"
    cv = KFold(n_splits=param_kfold)
    estimator = DecisionTreeClassifier(ccp_alpha=param_alpha, random_state=RANDOM_STATE)
    
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,0], ylim=(0.6, 1.01), cv=cv, n_jobs=4
    )
    #DTC testing results
    estimator.fit(X_train,Y_train)
    display_metrics("Decision Tree",estimator.predict(X_test),Y_test)
    
    
    #Boosted DTC
    param_alpha = 0.02 #complexity parameter alpha for Boosted decision tree
    title = "Learning Curves Boosted Decision Tree"
    cv = KFold(n_splits=param_kfold)
    DTC = DecisionTreeClassifier(ccp_alpha=param_alpha, random_state=RANDOM_STATE)
    estimator = AdaBoostClassifier(base_estimator=DTC, n_estimators=100)
    plot_learning_curve(
        estimator, title, X_train, Y_train, axes=axes[:,1], ylim=(0.6, 1.01), cv=cv, n_jobs=4
    )
    #Boosted DTC testing results
    estimator.fit(X_train,Y_train)
    display_metrics("Boosted Decision Tree",estimator.predict(X_test),Y_test) 
    
    plt.show()
	

if __name__ == "__main__":
	main()
	
