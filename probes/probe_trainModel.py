import torch
import torch.nn as nn
from torch.nn import MSELoss

from probe import OneWordPSDProbe
from constructLabel import ConstructLabelGaget
#from probe_loss import buff_Loss

class probe(nn.Module):
    def __init__(self, args):
        super(probe, self).__init__()
        self.oneWordpsdProbe = OneWordPSDProbe(args={'probe': {'maximum_rank': args['probe']['maximum_rank']}, 'model': {'hidden_dim': args['model']['hidden_dim']}})
        self.constructLabel = ConstructLabelGaget(args=None)
    
    def forward(self, batch):
        norms = self.oneWordpsdProbe(batch).to("cuda:0" if torch.cuda.is_available() else "cpu")
        pseu_labels = self.constructLabel(norms).to("cuda:0" if torch.cuda.is_available() else "cpu")

        #prob_loss = None
        #Prob_loss_fct = buff_Loss()
        #prob_loss = Prob_loss_fct(norms, pseu_labels)
        prob_loss = None
        Prob_loss_fct = MSELoss()
        prob_loss = Prob_loss_fct(norms, pseu_labels)
        
        return prob_loss
