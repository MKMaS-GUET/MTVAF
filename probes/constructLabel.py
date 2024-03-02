import torch
import torch.nn as nn

import math

class ConstructLabelGaget(nn.Module):
    def __init__(self, args):
        super(ConstructLabelGaget, self).__init__()
        #self.bias = nn.Parameter(data=torch.tensor(0.5))
    
    def forward(self, norms):
        batchlen, seqlen = norms.size()
        batchlabel = torch.zeros(batchlen, seqlen)
        for i in range(batchlen):
            sort_idx = [[j, norms[i][j]] for j in range(seqlen)]
            sort_idx.sort(key=lambda x: x[1])
            if seqlen >= 1:
                sort_idx[0][1] = 1
            if seqlen >= 2:
                sort_idx[1][1] = 2
            for j in range(2, seqlen):
                if(abs(sort_idx[j][1] - sort_idx[j - 1][1]) < abs(sort_idx[j - 1][1] + 1 - sort_idx[j][1])):
                    sort_idx[j][1] = sort_idx[j - 1][1]
                else:
                    sort_idx[j][1] = sort_idx[j - 1][1] + 1
            sort_idx.sort(key=lambda x: x[0])
            for j in range(seqlen):
                batchlabel[i][j] = sort_idx[j][1]
        return batchlabel
