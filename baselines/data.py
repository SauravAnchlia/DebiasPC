from fairlearn.metrics import (
    MetricFrame, plot_model_comparison,
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    false_positive_rate, false_negative_rate,
    false_positive_rate_difference, false_negative_rate_difference,
    equalized_odds_difference)

import json
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import argparse
import os
import math

DATA2D = {'adult': 'target',
          'compas': 'ScoreText_',
          'german': 'loan_status',
          'synthetic' : 'D'}

DATA2S = {'adult': 'sex',
          'compas': 'Ethnic_Code_Text_',
          'german': 'sex',
          'synthetic': 'S'}

NAMES = ['adult', 'compas', 'german', 'synthetic']

def read_config(name, exp_num):
    config_path = os.path.join("..", "exp", name, str(exp_num),"config.json")
    with open(config_path, 'r') as cfh:
        content = json.load(cfh)
    return content

def process_pred(pred, threshold):
    pred  = (pred[~np.isnan(pred)] > threshold).astype(int)
    return pred

def load_data(name, fold, num_X=None, use_fair=False, exp_num = None):
    if name == "synthetic":
        filename = '../data/synthetic_data/%s.csv' % num_X
        train_splits = '../data/splited_data/10-cv/synthetic/%s/train_ids.csv' % num_X
        test_splits = '../data/splited_data/10-cv/synthetic/%s/test_ids.csv' % num_X
    else:
        assert(name in NAMES)
        filename = '../data/processed_data/%s_binerized.csv' % name
        train_splits = '../data/splited_data/10-cv/%s/train_ids.csv' % name
        test_splits = '../data/splited_data/10-cv/%s/test_ids.csv' % name
    fair_labels = None
    if exp_num is not None:
        exp_num = str(exp_num)
        exp_config = read_config(name, exp_num)
        experiment_path =  os.path.join("..", "exp", name, exp_num, "para", "max-ll","predict-per-example-proxy-label.csv")
        # marginalize = exp_config["debias"]
        fair_labels = pd.read_csv(experiment_path)
    train_id = np.array(pd.read_csv(train_splits)['x%s' % fold]) - 1
    test_id = np.array(pd.read_csv(test_splits)['x%s' % fold]) - 1

    train_id = train_id[~np.isnan(train_id)].astype(int)
    test_id = test_id[~np.isnan(test_id)].astype(int)

    cloumns = [DATA2D[name], 'Df'] if name == "synthetic" else [DATA2D[name]]
    decision_label = 'Df' if (use_fair and name =="synthetic") else DATA2D[name]

    train_data = pd.read_csv(filename).iloc[train_id, :]
    test_data = pd.read_csv(filename).iloc[test_id, :]

    

    if name == "synthetic":
        test_y_fair = np.array(process_pred(fair_labels["P(Df|e) test_x"]), 0,5) if fair_labels is not None else np.array(test_data['Df'])
        test_y_proxy = np.array(test_data['D'])
        train_y_fair = np.array(process_pred(fair_labels["P(Df|e) train_x"]), 0,5) if fair_labels is not None else np.array(train_data['Df'])
        train_y_proxy = np.array(train_data['D'])
    else:
        test_y_fair = process_pred(fair_labels["P(Df|e) test_x"], 0.5) if fair_labels is not None else None
        test_y_proxy = np.array(test_data[DATA2D[name]])
        train_y_fair =  process_pred(fair_labels["P(Df|e) train_x"], 0.5) if fair_labels is not None else None
        train_y_proxy = np.array(train_data[DATA2D[name]])
    # if name == "synthetic":
    #     test_y_fair = np.array(test_data['Df'])
    #     test_y_proxy = np.array(test_data['D'])
    #     train_y_fair = np.array(train_data['Df'])
    #     train_y_proxy = np.array(train_data['D'])
    # else:
    #     test_y_fair = None
    #     test_y_proxy = np.array(test_data[DATA2D[name]])
    #     train_y_fair = None
    #     train_y_proxy = np.array(train_data[DATA2D[name]])


    return train_data, test_data, cloumns, decision_label, train_y_fair, train_y_proxy, test_y_fair, test_y_proxy


def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Dataset name", type=str)
    parser.add_argument("--num_X", help="Number of non sensitive variables for synthetic dataset", type=int, default=-1)
    parser.add_argument("--fold", help="Dataset split fold", type=int)
    parser.add_argument("--use_fair", help="Whether use fair label as decision label", type=bool, default=False)
    parser.add_argument("--exp_num", help="Experiment number to pick fair labels from", type=int)
    args = parser.parse_args()
    name = args.name
    fold = args.fold
    num_X = args.num_X
    use_fair = args.use_fair
    return name, fold, num_X, use_fair, args.exp_num

def save_summary(file_path, results):
    try:
        if not os.path.exists(file_path):  
            os.makedirs(file_path, exist_ok=True)
        
        file_name = os.path.join(file_path,  f"summary.csv")
        results.to_csv(file_name)
        return True    
    except Exception as e:
        print(f"Error: \n {e} \n encountered.")
        return False

def save_file(name, num_X, fold, outdir, prob_train, s_train, train_y_fair, train_y_proxy, prob_test, s_test, test_y_fair, test_y_proxy, exp_num = None, exp_name = None):

    # train 
    if name == "synthetic":
        df = pd.DataFrame(data={"prob": prob_train, "S_label": s_train, "true_label_fair": train_y_fair, "true_label_proxy": train_y_proxy})
    else:
        df = pd.DataFrame(data={"prob": prob_train, "S_label": s_train, "true_label_proxy": train_y_proxy})
    if exp_num is not None:
        outdir = os.path.join("..", "exp", name, str(exp_num), exp_name)
    else:
        outdir = 'exp-results/' + outdir +'/synthetic/%s/' % num_X if name == "synthetic" else 'exp-results/' + outdir + '/%s/' % name

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    df.to_csv(os.path.join(outdir,str(fold) + '-train.csv'))

    # test
    if name == "synthetic":
        df = pd.DataFrame(data={"prob": prob_test, "S_label": s_test, "true_label_fair":test_y_fair, "true_label_proxy": test_y_proxy})
    else:
        df = pd.DataFrame(data={"prob": prob_test, "S_label": s_test, "true_label_proxy": test_y_proxy})
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    df.to_csv(os.path.join(outdir,str(fold) + '-test.csv'))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def accuracy(prob, true_label, sv = None):
    predicted = prob > 0.5
    true_label = true_label.astype(bool)
    acc = sum(true_label == predicted) / float(true_label.size)
    return acc

def precision(prob, true_label, sv = None):
    predicted = prob > 0.5
    true_label = true_label.astype(bool)
    precision = (predicted[(true_label == True) & (predicted == 1)]).size/float(predicted[predicted == 1].size)
    # precision = sum(predicted.loc[true_label == True] == 1)/ float(predicted.loc[predicted == 1].size)
    return precision

def recall(prob, true_label, sv = None):
    predicted = prob > 0.5
    true_label = true_label.astype(bool)
    recall = (predicted[(predicted == 1) & (true_label == True)]).size/float(true_label[true_label == True].size)
    # recall = sum(predicted.loc[true_label == True] == 1)/ float(true_label.loc[true_label == True].size)
    return recall

def f1_score(precision, recall):
    return float(2 * float((precision * recall)/(precision + recall)))

def summary(models_dict, sv):
    res = {}
    for exp, (prob_train, prob_test, train_y, test_y) in models_dict.items():
        res[exp] = summary(prob_test, test_y, sv)
    return res
    

def summary(prob, true_label, sv):
    pred = (prob > 0.5).astype(int)
    fair_metrics = {}
    fair_metrics["accuracy"] = accuracy(prob, true_label)
    fair_metrics["precision"] = precision(prob, true_label)
    fair_metrics["f1_score"] = skm.f1_score(pred, true_label)
    fair_metrics["recall"] = recall(prob, true_label)
    fair_metrics["Overall selection rate"] = selection_rate(true_label, pred)
    fair_metrics["Demographic parity difference"] = demographic_parity_difference(true_label, pred, sensitive_features=sv)
    fair_metrics["Demographic parity ratio"] = demographic_parity_ratio(true_label, pred, sensitive_features=sv)
    fair_metrics["False positive rate difference"] = false_positive_rate_difference(true_label, pred, sensitive_features=sv)
    fair_metrics["Equalized odds difference"] = equalized_odds_difference(true_label, pred, sensitive_features=sv)
    fair_metrics["False negative rate difference"] = false_negative_rate_difference(true_label, pred, sensitive_features=sv)
    # print(fair_metrics)
    return fair_metrics
    # return pd.DataFrame.from_dict(fair_metrics, orient="index", columns=fair_metrics.keys())