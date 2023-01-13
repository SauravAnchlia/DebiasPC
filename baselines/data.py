import numpy as np
import pandas as pd
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

def process_train(pred, threshold):
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
    train_id = np.array(pd.read_csv(train_splits)['x%s' % fold]) - 1
    test_id = np.array(pd.read_csv(test_splits)['x%s' % fold]) - 1

    train_id = train_id[~np.isnan(train_id)].astype(int)
    test_id = test_id[~np.isnan(test_id)].astype(int)

    # read experiment specific paths
    if exp_num is not None:
        experiment_path =  os.path.join("..", "exp", name, exp_num, "para", "max-ll","predict-per-example-proxy-label.csv")
        exp_config_path =  os.path.join("..", "exp", name, exp_num, "config.json")

    cloumns = [DATA2D[name], 'Df']
    decision_label = 'Df' if (use_fair) else DATA2D[name]

    train_data = pd.read_csv(filename).iloc[train_id, :]
    test_data = pd.read_csv(filename).iloc[test_id, :]

    if use_fair:
        fair_labels = pd.read_csv(experiment_path)
        train_y_debias = process_train(fair_labels["P(Df|e) train_x"], 0.5)
        test_y_debias = process_train(fair_labels["P(Df|e) test_x"], 0.5)

    if name == "synthetic":
        test_y_fair = np.array(test_data['Df'])
        test_y_proxy = np.array(test_data['D'])
        train_y_fair = np.array(train_data['Df'])
        train_y_proxy = np.array(train_data['D'])
    else:
        test_y_fair = None
        test_y_proxy = np.array(test_data[DATA2D[name]])
        train_y_fair = None
        train_y_proxy = np.array(train_data[DATA2D[name]])


    return train_data, test_data, cloumns, decision_label, train_y_fair, train_y_proxy, test_y_fair, test_y_proxy, test_y_debias, train_y_debias


def read_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Dataset name", type=str)
    parser.add_argument("--num_X", help="Number of non sensitive variables for synthetic dataset", type=int, default=-1)
    parser.add_argument("--fold", help="Dataset split fold", type=int)
    parser.add_argument("--use_fair", help="Whether use fair label as decision label", type=bool, default=False)
    parser.add_argument("--exp_num", help="Provide experiment number to source fair label data", type = int, default=False)
    args = parser.parse_args()
    return args.name, args.fold, args.num_X, args.use_fair, args.exp_num


def save_file(name, num_X, fold, outdir, prob_train, s_train, train_y_fair, train_y_proxy, prob_test, s_test, test_y_fair, test_y_proxy):
    # train 
    if name == "synthetic":
        df = pd.DataFrame(data={"prob": prob_train, "S_label": s_train, "true_label_fair": train_y_fair, "true_label_proxy": train_y_proxy})
    else:
        df = pd.DataFrame(data={"prob": prob_train, "S_label": s_train, "true_label_proxy": train_y_proxy})
    
    outdir = 'exp-results/' + outdir +'/synthetic/%s/' % num_X if name == "synthetic" else 'exp-results/' + outdir + '/%s/' % name

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    df.to_csv(outdir + str(fold) + '-train.csv')

    # test
    if name == "synthetic":
        df = pd.DataFrame(data={"prob": prob_test, "S_label": s_test, "true_label_fair":test_y_fair, "true_label_proxy": test_y_proxy})
    else:
        df = pd.DataFrame(data={"prob": prob_test, "S_label": s_test, "true_label_proxy": test_y_proxy})
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    df.to_csv(outdir + str(fold) + '-test.csv')


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def accuracy(prob, true_label):
    predicted = prob > 0.5
    true_label = true_label.astype(bool)
    acc = sum(true_label == predicted) / float(true_label.size)
    return acc