import torch
from torch import nn

class CombineLoss(nn.Module):
    def __init__(self, para):
        super(CombineLoss, self).__init__()
        self.superParameter = torch.tensor(para)
    """
    def forward(self, loss, cola_loss, epoch):
        return loss + cola_loss * self.superParameter
    """

    def forward(self, loss, probe_loss, epoch):
        if(probe_loss.item() > 0.1):
            # print("loss:{},probe:{},*:{}".format(loss,probe_loss,probe_loss * self.superParameter * torch.tensor(pow(1.1, -epoch))))
            return loss + probe_loss * self.superParameter * torch.tensor(pow(2, -epoch))
        else:
            return loss

