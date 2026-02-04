from logging import getLogger
from .abstract_evaluator import AbstractEvaluator
import json
import pandas as pd
from sklearn import linear_model, svm
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import argparse
import os
import random


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def compute_mrr(true_labels, machine_preds):
    """Compute the MRR """
    rr_total = 0.0
    for i in range(len(true_labels)):
        ranklist = list(np.argsort(machine_preds[i])[::-1])
        rank = ranklist.index(true_labels[i]) + 1
        rr_total = rr_total + 1.0 / rank
    mrr = rr_total / len(true_labels)
    return mrr

def MAPE(labels, predicts, mask):
    loss = np.abs(predicts-labels) / (np.abs(labels))
    loss *= mask
    non_zero_len = mask.sum()
    return np.sum(loss)/non_zero_len


def metrics_local(y_truths, y_preds):
    y_preds[y_preds < 0] = 0

    mae = mean_absolute_error(y_truths, y_preds)
    rmse = np.sqrt(mean_squared_error(y_truths, y_preds))

    real_y_true_mask = (1 - (y_truths == 0))
    mape = MAPE(y_truths, y_preds, real_y_true_mask)

    return mae, rmse, mape


def scale_transform(f, min_o, max_o, min_t, max_t):
    transformed_values = (f-min_o)/(max_o-min_o)*(max_t-min_t) + min_t
    return transformed_values


def evaluation_reg(s_X, s_y, t_X, t_y, seed=42, output_dim=128):
    X_train, X_test = s_X, t_X
    y_train, y_test = s_y, t_y

    reg = linear_model.Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_truth = y_test

    mae, rmse, mape = metrics_local(y_truth, y_pred)
    return mae, rmse, mape


class Evaluator(object):
    def __init__(self, exp_id, seed=42, output_dim=128):
        self.result = {}
        # self.dataset = config.get('dataset', '')
        self.s_dataset = None
        self.t_dataset = None
        self.exp_id = exp_id
        self.cluster_kinds = 5
        self.seed = seed
        self.data_path = './raw_data/'
        self.s_data_path = None
        self.t_data_path = None
        self.output_dim = output_dim
        self.best_epoch = None
        # self.regionid = config.get('regionid', None)
        self.region_embedding_path = './output/{}/train_cache/'.format(self.exp_id)

    def collect(self, batch):
        pass

    def _valid_cross_region_gdp(self, s_region_emb, t_region_emb):
        print('Evaluating Region GDP Prediction')
        s_gdp = np.load(self.data_path + '{}/gdp_{}_sum.npy'.format(
            self.s_dataset, self.s_dataset)).astype('float32')
        t_gdp = np.load(self.data_path + '{}/gdp_{}_sum.npy'.format(
            self.t_dataset, self.t_dataset)).astype('float32')

        mae, rmse, mape = evaluation_reg(s_region_emb, s_gdp, t_region_emb, t_gdp)
        print("Result of {} estimation in {} based on {}:".format('GDP', self.t_dataset, self.s_dataset))
        print('MAE = {:.3f}, RMSE = {:.3f}, MAPE = {:.3f}'.format(mae, rmse, mape))
        return mae, rmse, mape

    def _valid_cross_region_pp(self, s_region_emb, t_region_emb):
        print('Evaluating Region Population Prediction')
        s_gdp = np.load(self.data_path + '{}/population_{}_sum.npy'.format(
            self.s_dataset, self.s_dataset)).astype('float32')
        t_gdp = np.load(self.data_path + '{}/population_{}_sum.npy'.format(
            self.t_dataset, self.t_dataset)).astype('float32')

        mae, rmse, mape= evaluation_reg(s_region_emb, s_gdp, t_region_emb, t_gdp)
        print("Result of {} estimation in {} based on {}:".format('population', self.t_dataset, self.s_dataset))
        print('MAE = {:.3f}, RMSE = {:.3f}, MAPE = {:.3f}'.format(mae, rmse, mape))
        return mae, rmse, mape

    def _valid_cross_region_co2(self, s_region_emb, t_region_emb):
        print('Evaluating Region CO2 Prediction')
        s_gdp = np.load(self.data_path + '{}/co2_{}_sum.npy'.format(
            self.s_dataset, self.s_dataset)).astype('float32')
        t_gdp = np.load(self.data_path + '{}/co2_{}_sum.npy'.format(
            self.t_dataset, self.t_dataset)).astype('float32')

        mae, rmse, mape = evaluation_reg(s_region_emb, s_gdp, t_region_emb, t_gdp)
        print("Result of {} estimation in {} based on {}:".format('CO2', self.t_dataset, self.s_dataset))
        print('MAE = {:.3f}, RMSE = {:.3f}, MAPE = {:.3f}'.format(mae, rmse, mape))
        return mae, rmse, mape


    def evaluate_region_embedding(self, s_region_emb, t_region_emb):
        print('Load source regin emb {}, source region emb shape = {}, target region shape = {}'.format(
            self.region_embedding_path, s_region_emb.shape, t_region_emb.shape))

        gdp_mae, gdp_rmse, gdp_mape = self._valid_cross_region_gdp(s_region_emb, t_region_emb)
        pp_mae, pp_rmse, pp_mape = self._valid_cross_region_pp(s_region_emb, t_region_emb)
        co2_mae, co2_rmse, co2_mape = self._valid_cross_region_co2(s_region_emb, t_region_emb)

        self.result['gdp_mae'] = gdp_mae
        self.result['gdp_rmse'] = gdp_rmse
        self.result['gdp_mape'] = gdp_mape

        self.result['pp_mae'] = pp_mae
        self.result['pp_rmse'] = pp_rmse
        self.result['pp_mape'] = pp_mape

        self.result['co2_mae'] = co2_mae
        self.result['co2_rmse'] = co2_rmse
        self.result['co2_mape'] = co2_mape

    def evaluate(self, s_dataset, t_dataset, s_region_emb, t_region_emb):
        self.s_dataset = s_dataset
        self.t_dataset = t_dataset
        self.s_data_path = self.data_path + self.s_dataset + '/'
        self.t_data_path = self.data_path + self.t_dataset + '/'
        self.evaluate_region_embedding(s_region_emb, t_region_emb)

        return self.result

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        self.result = {}


