from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings


def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])


def profit_curve(cost_benefit, predicted_probs, labels):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and thier true labels.

    Parameters
    ----------
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1

    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """
    n_obs = float(len(labels))
    # Make sure that 1 is going to be one of our thresholds
    maybe_one = [] if 1 in predicted_probs else [1]
    thresholds = maybe_one + sorted(predicted_probs, reverse=True)
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / n_obs
        profits.append(threshold_profit)
    return np.array(profits), np.array(thresholds)


def get_model_profits(model, cost_benefit, X_train, X_test, y_train, y_test):
    """Fits passed model on training data and calculates profit from cost-benefit
    matrix at each probability threshold.

    Parameters
    ----------
    model           : sklearn model - fitted but to be predict_proba
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    X_train         : ndarray - 2D
    X_test          : ndarray - 2D
    y_train         : ndarray - 1D
    y_test          : ndarray - 1D

    Returns
    -------
    model_profits : model, profits, thresholds
    """
    #model.fit(X_train, y_train)
    predicted_probs = model.predict_proba(X_test)[:, 1]
    profits, thresholds = profit_curve(cost_benefit, predicted_probs, y_test)

    return profits, thresholds


def plot_model_profits(model_profits, save_path=None):
    """Plotting function to compare profit curves of different models.

    Parameters
    ----------
    model_profits : list((model, profits, thresholds))
    save_path     : str, file path to save the plot to. If provided plot will be
                         saved and not shown.
    """
    for model, profits, _ in model_profits:
        percentages = np.linspace(0, 100, profits.shape[0])
        plt.plot(percentages, profits, label=model.__class__.__name__)

    plt.title("Profit Curves")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def find_best_threshold(model_profits):
    """Find model-threshold combo that yields highest profit.

    Parameters
    ----------
    model_profits : list((model, profits, thresholds))

    Returns
    -------
    max_model     : str
    max_threshold : float
    max_profit    : float
    """
    max_model = None
    max_threshold = None
    max_profit = None
    for model, profits, thresholds in model_profits:
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit


def profit_curve_main(fitted_models, X_train, X_test, y_train, y_test, cost_benefit):
    """Main function to test profit curve code.

    Parameters
    ----------
    filepath     : str - path to find churn.csv
    cost_benefit  : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    """
    #X_train, X_test, y_train, y_test = get_train_test(filepath)
    #models = [RF(), LR(), GBC(), SVC(probability=True),KNC()]
    model_profits = []
    for model in fitted_models:
        profits, thresholds = get_model_profits(model, cost_benefit,
                                                X_train, X_test,
                                                y_train, y_test)
    model_profits.append((model, profits, thresholds))
    plot_model_profits(model_profits)
    max_model, max_thresh, max_profit = find_best_threshold(model_profits)
    max_labeled_positives = max_model.predict_proba(X_test) >= max_thresh
    proportion_positives = max_labeled_positives.mean()
    reporting_string = ('Best model:\t\t{}\n'
                        'Best threshold:\t\t{:.2f}\n'
                        'Resulting profit:\t{}\n'
                        'Proportion positives:\t{:.2f}')
    print(reporting_string.format(max_model.__class__.__name__, max_thresh,
                                  max_profit, proportion_positives))


def div_count_pos_neg(X, y):
    """Helper function to divide X & y into positive and negative classes
    and counts up the number in each.

    Parameters
    ----------
    X : ndarray - 2D
    y : ndarray - 1D

    Returns
    -------
    negative_count : Int
    positive_count : Int
    X_positives    : ndarray - 2D
    X_negatives    : ndarray - 2D
    y_positives    : ndarray - 1D
    y_negatives    : ndarray - 1D
    """
    negatives, positives = y == 0, y == 1
    negative_count, positive_count = np.sum(negatives), np.sum(positives)
    X_positives, y_positives = X[positives], y[positives]
    X_negatives, y_negatives = X[negatives], y[negatives]
    return negative_count, positive_count, X_positives, \
           X_negatives, y_positives, y_negatives


def mini_test():
    test_cost_benefit = np.array([[6, -3], [0, 0]])
    test_probs = np.array([0.2, 0.6, 0.4])
    test_labels = np.array([0, 0, 1])

    profit_curve_results = profit_curve(test_cost_benefit, test_probs, test_labels)
    print(profit_curve_results)
    for profit, threshold in zip(*profit_curve_results):
        print('Profit at threshold of {}: {}'.format(threshold, profit))

def base_KNC(X_train,y_train,X_test):
    scores = np.zeros(10)
    for n in range(2,9):
        knc = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
        knc.fit(X_train, y_train)
        pred = knc.predict(X_test)
        pred_prob = knc.predict_proba(X_test)
        #knc.kneighbors_graph(X_train,5)
        accuracy = knc.score(X_train,y_train)
        precision = ps(y_test,pred)
        cross_val_score(knc, X_train, y_train,scoring='precision',cv=5)
        scores[n]=precision
    return scores

def work():
    cost_benefit = np.array([[100, -10], [ 0,   0]])
    knc = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
    knc.fit(X_train, y_train)

    fitted_models = [knc]
    profit_curve_main(fitted_models, X_train, X_test, y_train, y_test, cost_benefit)
    #EDAandCleaning()


if __name__ == '__main__':
    pass
