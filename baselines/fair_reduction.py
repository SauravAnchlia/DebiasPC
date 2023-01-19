import numpy as np
import pandas as pd
import argparse
import sys
import os
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity, EqualizedOdds
from data import *
import fair_classification.utils as ut


def reduction(name, fold, num_X=0, use_debias = False, exp_num = None):
    # load data
    train_data, test_data, cloumns, decision_label, train_y_fair, train_y_proxy, test_y_fair, test_y_proxy , test_y_debias, train_y_debias= load_data(name, fold, num_X=num_X, use_fair=use_debias, exp_num = exp_num)
    # drp[s Data2D[name] value due to presence in columns from load_data
    train_y_true = None
    train_X = train_data.drop(columns=cloumns)
    if use_debias and name != "synthetic":
        train_y_true = train_y_fair
    elif name == "synthetic" and use_debias:
        train_y_true = train_y_debias
    else:
        train_y_true = train_data[decision_label]
    # train_y_true = train_data[DATA2D[name]]
    train_sex = train_data[DATA2S[name]]

    # learn
    learn = ExponentiatedGradient(
        LogisticRegression(solver='liblinear', fit_intercept=True),
        constraints=EqualizedOdds())
    learn.fit(train_X, train_y_true, sensitive_features=train_sex)

    # predict
    test_X = test_data.drop(columns=cloumns)
    prob_test = learn._pmf_predict(test_X)[:, 1]
    # prob_test = (prob_test > 0.5).astype(int)
    prob_train = learn._pmf_predict(train_X)[:, 1]
    # prob_train = (prob_train > 0.5).astype(int)

    s_train = train_sex.astype(bool)
    s_test = np.array(test_data[DATA2S[name]]).astype(bool)
    test_y = None
    if use_debias and name != "synthetic":
        test_y = test_y_fair
    elif name == "synthetic" and use_debias:
        test_y = test_y_debias
    else:
        test_y = test_data[decision_label]
    
    print("Utils: check accuracy score")
    print(ut.check_accuracy(None, train_X, train_y_true, test_X, test_y, prob_train, prob_test))
    print("Equalized odds values:")
    print(ut.get_equalized_odds_difference(test_y, np.array(prob_test >0.5).astype(int), s_test))
    # print(ut.calculate_equal_opportunity(test_y, prob_test, s_test))
    save_file(name, num_X, fold, "Reduction", prob_train, s_train, train_y_fair, train_y_proxy, prob_test, s_test, test_y_fair, test_y_proxy, use_debias)
    return prob_train, prob_test

def main():
    name, fold, num_X, use_fair, exp_num= read_cmd()
    config = read_config(name, exp_num)
    reduction(name, fold, num_X=num_X, use_debias = use_fair, exp_num=exp_num)


if __name__ == "__main__":
    main()
