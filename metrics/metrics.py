from dtw import dtw,accelerated_dtw
import torch

def dwt(first_signal, second_signal):
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(first_signal,second_signal, dist='euclidean')
    return d

from einops import rearrange
from scipy.stats import pearsonr

def PCC(first_signal, second_signal):
    return pearsonr(first_signal, second_signal)[0] # the first value is the correlation coefficient

def pairwise_mse(first_signal, second_signal, mean_dim=0):
    return ((first_signal - second_signal)**2).mean(mean_dim)

from numpy import mean


def painsim(pspi_predict, pspi_gt):
    b, t = pspi_predict.shape
        
    dwt_list = []
    for sample_id in range(b):
        
        dwt_list.append(dwt(pspi_predict[sample_id], pspi_gt[sample_id]))
        
    return mean(dwt_list)


def paincorr(pspi_predict, pspi_gt):
    b, t = pspi_predict.shape
        
    pcc_list = []
    for sample_id in range(b):
        
        pcc_list.append(PCC(pspi_predict[sample_id], pspi_gt[sample_id]))
        
    return max(pcc_list)

def painacc(pspi_predict, stimuli):
    b, t = pspi_predict.shape
    
    acc_list = []
    for sample_id in range(b):
        
        acc_list.append(dwt(pspi_predict[sample_id], stimuli[sample_id]))
        
    return mean(acc_list)

def paindist(exp_pred, exp_gt):
    b, t, d = exp_pred.shape
    
    mse_list = []
    for sample_id in range(b):
        
        mse_list.append(pairwise_mse(exp_pred[sample_id], exp_gt[sample_id]))
        
    return mean(mse_list)

def paindivrs(preds):     
    # preds: (10, B, 750, 25) 
    preds_ = rearrange(preds, 'n b t d -> b n (t d)').contiguous()
    # preds_: (10, B, 750*25)
    dist = torch.pow(torch.cdist(preds_, preds_), 2)
    # dist: (10, B, B)
    dist = torch.sum(dist) / (preds.shape[0] * (preds.shape[0] - 1) * preds.shape[1])
    return dist / preds_.shape[-1]

def painvar(preds):
    # preds: (B, T, D)
    var = torch.var(preds, dim=1)
    return torch.mean(var)


import numpy as np


def error_accumulation(exp_pred, exp_gt):
    
    # exp_pred: (B, T, D)
    
    b, t, d = exp_pred.shape
    
    pcc_list = []
    
    for b_id in range(b):
        
        pairwise_error = pairwise_mse(exp_pred[b_id], exp_gt[b_id], mean_dim=1)
        
        timestep = torch.arange(exp_pred[b_id].shape[0])
        
        # print(pairwise_error.shape, timestep.shape)
        pcc_of_timestep = PCC(pairwise_error, timestep)
        
        pcc_list.append(pcc_of_timestep)
        
    # filter nan values
    pcc_list = [x for x in pcc_list if not np.isnan(x)]
        
    return mean(pcc_list)


def calculate_pain_metrics(exp_pred, exp_multiple, exp_gt, pspi_pred, pspi_gt, stimuli):
    
        # print("exp_pred", exp_pred.shape)
        # print("exp_multiple", exp_multiple.shape)
        # print("exp_gt", exp_gt.shape)
        # print("pspi_pred", pspi_pred.shape)
        # print("pspi_gt", pspi_gt.shape)
        # print("stimuli", stimuli.shape)
        
        pain_dist = paindist(exp_pred, exp_gt)
        print("pain_dist", pain_dist)
        pain_divrs = paindivrs(exp_multiple)
        print("pain_divrs", pain_divrs)
        pain_var = painvar(exp_pred)
        print("pain_var", pain_var)
        pain_corr = paincorr(pspi_pred, pspi_gt)
        print("pain_corr", pain_corr)
        pain_sim = painsim(pspi_pred, pspi_gt)
        print("pain_sim", pain_sim)
        # error_accum = error_accumulation(exp_pred, exp_gt)
        # print("error_accum", error_accum)
        pain_acc = painacc(pspi_pred, stimuli)
        print("pain_acc", pain_acc)
        
        return (
            pain_dist, 
                pain_divrs, 
                pain_var, 
                pain_corr, 
                pain_sim, 
                # error_accum, 
                pain_acc
                )

class RunningMean():
    def __init__(self) -> None:
        
        self.list = []
        
    def add(self, value):
        self.list.append(value)
        
    def mean(self):
        
        return np.mean(self.list)
    

class Metrics():
    
    def __init__(self) -> None:
        self.pain_dist = RunningMean()
        self.pain_divrs = RunningMean()
        self.pain_var = RunningMean()
        self.pain_corr = RunningMean()
        self.pain_sim = RunningMean()
        self.error_accum = RunningMean()
        self.pain_acc = RunningMean()
        
    def add(self, pain_dist, pain_divrs, pain_var, pain_corr, pain_sim, error_accum, pain_acc):
        
        self.pain_dist.add(pain_dist)
        self.pain_divrs.add(pain_divrs)
        self.pain_var.add(pain_var)
        self.pain_corr.add(pain_corr)
        self.pain_sim.add(pain_sim)
        self.error_accum.add(error_accum)
        self.pain_acc.add(pain_acc)
        
    def calculate(self, exp_pred, exp_multiple, exp_gt, pspi_pred, pspi_gt, stimuli):
        
        pain_dist, pain_divrs, pain_var, pain_corr, pain_sim, error_accum, pain_acc = calculate_pain_metrics(exp_pred, exp_multiple, exp_gt, pspi_pred, pspi_gt, stimuli)
        
        self.add(pain_dist, pain_divrs, pain_var, pain_corr, pain_sim, error_accum, pain_acc)
        
    def get(self):
        
        return self.pain_dist.mean(), self.pain_divrs.mean(), self.pain_var.mean(), self.pain_corr.mean(), self.pain_sim.mean(), self.error_accum.mean(), self.pain_acc.mean()