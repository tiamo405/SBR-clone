import os
# os['CUDA_VISIBLE_DEVICES']=1
import numpy as np
from proc_utils import Dataset, split_validation
from model import *

import pickle

def _load_file(filename):
  with open(filename, 'rb') as fn:
    data = pickle.load(fn)
  return data
test_data_set = _load_file('datasets/yoochoose_data_64/test.pkl')
test_data = Dataset(test_data_set, shuffle=False)

class yoochoose_data_64():
    dataset = 'yoochoose_data_64'
    batchSize = 1
    hiddenSize = 120
    epoch = 100
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 3
    l2 = 1e-5
    step = 1
    patience = 10
    nonhybrid = True
    validation = True
    valid_portion = 0.1
opt = yoochoose_data_64()
n_node = 22055
model = to_cpu(Attention_SessionGraph(opt, n_node))
model = torch.load('weights/epoch_3_recall_37.54314344034144_.pt',map_location=torch.device('cpu'))
model = model.eval()
model.to('cpu')
print(model)


hit, mrr = [], []
slices = test_data.generate_batch(model.batch_size)
print(slices[0].shape)

for i in slices:
    targets, scores = forward(model, i, test_data)
    sub_scores = scores.topk(20)[1]
    sub_scores = to_cpu(sub_scores).detach().numpy()

    for score, target, mask in zip(sub_scores, targets, test_data.mask):
        hit.append(np.isin(target - 1, score))
        if len(np.where(score == target - 1)[0]) == 0:
            mrr.append(0)
        else:
            mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

hit = np.mean(hit) * 100
mrr = np.mean(mrr) * 100

print(f'hit: {hit}, mrr: {mrr}')