import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
from data import *

def debias_weights(original_labels, protected_attributes, multipliers):
  exponents = np.zeros(len(original_labels))
  for i, m in enumerate(multipliers):
    exponents -= m * protected_attributes[i]
  weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
  weights = np.where(original_labels > 0, 1 - weights, weights)
  return weights

def get_error_and_violations(y_pred, y, protected_attributes):
  acc = np.mean(y_pred != y)
  violations = []
  for p in protected_attributes:
    protected_idxs = np.where(p > 0)
    violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
  pairwise_violations = []
  for i in range(len(protected_attributes)):
    for j in range(i+1, len(protected_attributes)):
      protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
      if len(protected_idxs[0]) == 0:
        continue
      pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
  return acc, violations, pairwise_violations


def learning(X_train, y_train, X_test, y_test, protected_train, protected_test):
    multipliers = np.zeros(len(protected_train))
    learning_rate = 1.
    n_iters = 100
    for it in range(n_iters):
        weights = debias_weights(y_train, protected_train, multipliers)
        model = LogisticRegression()

        model.fit(X_train, y_train, weights)
        y_pred_train = model.predict(X_train)
        acc, violations, pairwise_violations = get_error_and_violations(y_pred_train, y_train, protected_train)
        multipliers += learning_rate * np.array(violations)


    if (it + 1) % n_iters == 0:
        print(multipliers)
        y_pred_test = model.predict(X_test)
        test_scores = model.decision_function(X_test)
        train_scores = model.decision_function(X_train)
        prob_test = np.array([sigmoid(x) for x in test_scores])
        prob_train = np.array([sigmoid(x) for x in train_scores])

        train_acc, train_violations, train_pairwise_violations = get_error_and_violations(y_pred_train, y_train, protected_train)
        print("Train Accuracy", train_acc)
        print("Train Violation", max(np.abs(train_violations)), " \t\t All violations", train_violations)
        if len(train_pairwise_violations) > 0:
            print("Train Intersect Violations", max(np.abs(train_pairwise_violations)), " \t All violations", train_pairwise_violations)

        test_acc, test_violations, test_pairwise_violations = get_error_and_violations(y_pred_test, y_test, protected_test)
        print("Test Accuracy", test_acc)
        print("Test Violation", max(np.abs(test_violations)), " \t\t All violations", test_violations)
        if len(test_pairwise_violations) > 0:
            print("Test Intersect Violations", max(np.abs(test_pairwise_violations)), " \t All violations", test_pairwise_violations)
        print()
        print()
        res = {}
        res["prob_train"] = y_pred_train
        res["prob_test"] = y_pred_test
        res["train_acc"] = train_acc
        res["train_violations"] = max(np.abs(train_violations))
        res["train_all_violations"] = train_violations
        res["train_intersect_violations"] = max(np.abs(train_pairwise_violations)) if len(train_pairwise_violations) > 0 else None
        res["train_all_pairwise_violations"] = train_pairwise_violations  if len(train_pairwise_violations) > 0 else None
        res["test_acc"] = test_acc
        res["test_violations"] = max(np.abs(test_violations))
        res["test_all_violations"] = test_violations
        res["test_intersect_violations"] = max(np.abs(test_pairwise_violations))  if len(test_pairwise_violations) > 0 else None
        res["test_all_pairwise_violations"] = test_pairwise_violations  if len(test_pairwise_violations) > 0 else None
        return res
        # return prob_train, prob_test, train_acc, train_violations, train_pairwise_violations, test_acc, test_violations, test_pairwise_violations


def model(name, fold, num_X=None, use_fair = None, exp_num = None):
    train_data, test_data, cloumns, learn_decision_label, train_y_fair, train_y_proxy, test_y_fair, test_y_proxy = load_data(name, fold, num_X=num_X, use_fair=False, exp_num=exp_num)
    results = pd.DataFrame()
    # load_data
    X_train = np.array(train_data.drop(columns=cloumns))
    X_test = np.array(test_data.drop(columns=cloumns))
    # proxy label -> train, fair_label -> test
    # fair label -> train, fair_label -> test
    # fair label -> train, proxy label -> test
    
    s_train = np.array(train_data[DATA2S[name]])
    protected_train = [s_train]
    s_test = np.array(test_data[DATA2S[name]])
    protected_test = [s_test]

    exp_name = os.path.basename(__file__)
    pre_prob_train, pre_prob_test = learning(X_train, train_y_fair, X_test, test_y_proxy, protected_train,protected_test)
    post_prob_train, post_prob_test = learning(X_train, train_y_proxy, X_test, test_y_fair, protected_train, protected_test )
    pre_post_prob_train, pre_post_prob_test = learning(X_train, train_y_fair, X_test, test_y_fair, protected_train, protected_test)
    proxy_train, proxy_test = learning(X_train, train_y_proxy, X_test, test_y_proxy, protected_train, protected_test)
    models = {"pre":[pre_prob_train, pre_prob_test,train_y_fair, test_y_proxy], "post":[post_prob_train, post_prob_test,train_y_proxy, test_y_fair], "pre_post":[pre_post_prob_train, pre_post_prob_test, train_y_fair, test_y_fair], "proxy":[proxy_train, proxy_test, train_y_proxy, test_y_proxy]}
    # prob_train, prob_test = learning(X_train, y_train, X_test, y_test, protected_train, protected_test)
    fair_metrics = summary(models, s_test, s_test)
    print(fair_metrics)
    if exp_num is not None:
      save_summary(os.path.join("..", "exp", name , str(exp_num),exp_name),  pd.json_normalize(fair_metrics))
    else:
      save_summary(os.path.join("exp-results", "Reweight" , name),  pd.json_normalize(fair_metrics))

    #print(1 - accuracy(prob, true_y_proxy))
    # save_file(name, num_X, fold, "Reweight", prob_train, s_train, train_y_fair, train_y_proxy, prob_test, s_test, test_y_fair, test_y_proxy, exp_num=exp_num, exp_name= exp_name)


def main():
    name, fold, num_X, use_fair, exp_num  = read_cmd()
    # if exp_num does not exist, analysis should be stored in general folder under exp
    model(name, fold, num_X=num_X, use_fair=use_fair, exp_num = exp_num)


if __name__ == "__main__":
    main()