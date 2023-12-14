import torch
import torch.nn as nn
from utils.WSDistance_layers import SinkhornDistance

class segmentLoss(nn.Module):
    def __init__(self):
        super(segmentLoss, self).__init__()
    def forward(self, y_pred, y_true):
        smooth = 1e-10
        ce_loss = -(y_true * torch.log(y_pred + smooth) + (1 - y_true) * torch.log(1 - y_pred + smooth))
        all_ce_value = torch.mean(ce_loss)
        return all_ce_value

class boundaryLoss(nn.Module):
    def __init__(self):
        super(boundaryLoss, self).__init__()
    def forward(self, preds, targets):
        smooth = 1e-10
        ce_loss = -(targets*torch.log(preds+smooth)+(1-targets)*torch.log(1-preds+smooth))
        all_ce_value = torch.mean(ce_loss)
        return all_ce_value.cuda()

class DiveLoss(nn.Module):
    def __init__(self):
        super(DiveLoss, self).__init__()

    def forward(self, y_pred,b_pred):
        y_pred_mean = y_pred.mean(dim=(2,3))
        y_pred_mu = y_pred_mean.mean(0)
        y_pred_std= ((y_pred_mean-y_pred_mu)**2).mean(0,keepdim=True).clamp(min=1e-12).sqrt()
        y_pred_loss_std = -y_pred_std.sum()

        b_pred_mean = b_pred.mean(dim=(2, 3))
        b_pred_mu = b_pred_mean.mean(0)
        b_pred_std = ((b_pred_mean - b_pred_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()
        b_pred_loss_std = -b_pred_std.sum()
        loss_std = y_pred_loss_std + b_pred_loss_std
        return loss_std
    

class HarLoss(nn.Module):
    def __init__(self):
        super(HarLoss, self).__init__()

    def forward(self, y_pred, b_pred):
        # Diver loss
        y_pred_mean = y_pred.mean(dim=(2,3))
        y_pred_mu = y_pred_mean.mean(0)
        y_pred_std= ((y_pred_mean-y_pred_mu)**2).mean(0,keepdim=True).clamp(min=1e-12).sqrt()
        y_pred_loss_std = -y_pred_std.sum()

        b_pred_mean = b_pred.mean(dim=(2, 3))
        b_pred_mu = b_pred_mean.mean(0)
        b_pred_std = ((b_pred_mean - b_pred_mu) ** 2).mean(0, keepdim=True).clamp(min=1e-12).sqrt()
        b_pred_loss_std = -b_pred_std.sum()

        dive_std = y_pred_loss_std + b_pred_loss_std
       
        # Wass loss
        t_y_pred = torch.tensor(y_pred_mean, dtype=torch.float)
        t_b_pred = torch.tensor(y_pred_mean, dtype=torch.float)
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        dive_wass, P, C = sinkhorn(t_y_pred, t_b_pred)
        dive_wass = dive_wass.sum()

        return dive_std, dive_wass

class PHNet_ceLoss(nn.Module):
    def __init__(self, batch=True):
        super(PHNet_ceLoss, self).__init__()
        self.batch = batch
        self.cloud_loss = segmentLoss()
        self.bound_loss = boundaryLoss()

    def __call__(self, y_pred, y_true, b_pred, b_true):
        y_true = y_true.float()
        b_pred = b_pred.float()
        b_true = b_true.float()

        bloss = self.bound_loss(b_pred, b_true)
        CEloss = self.cloud_loss(y_pred, y_true)

        weight = 0.4
        loss =  weight*CEloss + (1-weight)*bloss
        print("segLoss:%f,         boundLoss:%f"%(10*CEloss.data,10*bloss.data))

        return loss,CEloss,bloss


class PHNet_divLoss(nn.Module):
    def __init__(self, batch=True):
        super(PHNet_divLoss, self).__init__()
        self.batch = batch
        self.cloud_loss = segmentLoss()
        self.bound_loss = boundaryLoss()
        self.diversity_loss = DiveLoss()

    def __call__(self, y_pred, y_true, b_pred, b_true):
        y_true = y_true.float()     # label is 0=clear, 128=unknown, 255=cloud
        b_pred = b_pred.float()
        b_true = b_true.float()
        bloss = self.bound_loss(b_pred, b_true)
        CEloss = self.cloud_loss(y_pred, y_true)
        diverseloss = self.diversity_loss(y_pred,b_pred)

        weight = 0.4
        loss =  weight*CEloss + (1-weight)*bloss + 0.1*diverseloss
        print("weight:%f,   segLoss:%f,         boundLoss:%f,         dive_STD:%f"%(weight, CEloss.data, bloss.data, diverseloss.data))

        return loss, CEloss, bloss, diverseloss


class PHNet_harLoss(nn.Module):
    def __init__(self, batch=True):
        super(PHNet_harLoss, self).__init__()
        self.batch = batch
        self.cloud_loss = segmentLoss()
        self.bound_loss = boundaryLoss()
        # self.diversity_loss = diversityLoss()
        self.HarLoss = HarLoss()

    def __call__(self, y_pred, y_true, b_pred, b_true):
        y_true = y_true.float()     # label is 0=clear, 128=unknown, 255=cloud
        b_pred = b_pred.float()
        b_true = b_true.float()

        bloss = self.bound_loss(b_pred, b_true)
        CEloss = self.cloud_loss(y_pred, y_true)
        dive_std, dive_wass = self.HarLoss(y_pred, b_pred)

        alpah, gamma = 0.4, 0.3

        loss = alpah * CEloss + (1 - alpah) * bloss + 0.1 * (gamma * dive_std +  (1 - gamma) * dive_wass)
        print("weight:%f,   segLoss:%f,   boundLoss:%f,    dive_std:%f,   dive_mean:%f   gamma:%f" %
              (alpah, CEloss.data, bloss.data, dive_std.data, dive_wass.data, gamma))

        return loss, CEloss, bloss, dive_std, dive_wass
