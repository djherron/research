#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:46:17 2022

@author: dave
"""

'''
Explore VRD evaluation metrics.

Recall@K, Avg Recall@K, Mean Avg Recall@K
'''

#%%

import pandas as pd

#%%

gt_vrs = [1, 3, 5, 7, 9]

# best case possible
pred_vrs_best = [5, 3, 1, 9, 7, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15]

# intermediate cases (ranked in descending order)
pred_vrs_int1 = [2, 3, 7, 1, 6, 5, 8, 10, 4, 11, 12, 9, 13, 14, 15]
pred_vrs_int2 = [2, 3, 4, 1, 6, 5, 8, 10, 7, 11, 12, 9, 13, 14, 15]
pred_vrs_int3 = [2, 3, 4, 1, 6, 5, 8, 10, 7, 11, 12, 13, 13, 9, 15]
pred_vrs_int4 = [2, 4, 6, 1, 6, 8, 10, 12, 7, 11, 13, 14, 9, 15, 16]
pred_vrs_int5 = [2, 4, 6, 8, 10, 12, 5, 13, 14, 15, 7, 16, 17, 1, 18]

# worst case possible
pred_vrs_worst = [2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


#%%

pred_vrs = pred_vrs_int3

#%%

def calc_recall_at_k(prd_vrs, gt_vrs):
    n_gt_vrs = len(gt_vrs)
    ks = []
    gt_vr_cnt_at_k = []
    recall_at_k = []   
    cnt = 0
    
    for idx in range(len(prd_vrs)):
        if prd_vrs[idx] in gt_vrs:
            cnt += 1        
        if idx >= n_gt_vrs-1:
            ks.append(idx+1)
            gt_vr_cnt_at_k.append(cnt)
            recall_at_k.append(round(gt_vr_cnt_at_k[-1] / n_gt_vrs, 5))

    df = pd.DataFrame({'k' : ks,
                       'gt_vr_cnt' : gt_vr_cnt_at_k,
                       'recall' : recall_at_k})
    
    return df
  
#%%

res = calc_recall_at_k(pred_vrs, gt_vrs)

print(res)

avg_recall_at_k = res.recall.sum() / res.shape[0]
avg_recall_at_k = round(avg_recall_at_k, 5)
print(f'Avg recall@K: {avg_recall_at_k}')

#%%

# extract recall_at_k at discrete levels of k

k = 10
row_mask = res.k == k
row_idx = res.k[row_mask].index.values.item()
recall_at_k = res.recall[row_idx]
print(f'recall @ k={k}: {recall_at_k}')

k = 15
row_mask = res.k == k
row_idx = res.k[row_mask].index.values.item()
recall_at_k = res.recall[row_idx]
print(f'recall @ k={k}: {recall_at_k}')








