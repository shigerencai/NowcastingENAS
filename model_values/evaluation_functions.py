import csv
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from core.utils import *


class SSIM(nn.Module):
    def __init__(self, window_size, img_channel, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = img_channel
        self.window = self.create_window(window_size,channel=self.channel).cuda()

    def forward(self, img1, img2, full=False, val_range=None):
        batch, seq, height, width, channel = img1.shape
        img1 = img1.permute(1, 0, 4, 2, 3).contiguous().reshape(seq*batch, channel, height, width)
        img2 = img2.permute(1, 0, 4, 2, 3).contiguous().reshape(seq*batch, channel, height, width)

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = self.create_window(self.window_size, channel).to(img1.device).type(img1.dtype).cuda()
            self.window = window
            self.channel = channel

        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if self.size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs

        return ret

    def create_window(self, window_size, channel):
        sigma = 1.5
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        _1D_window = (gauss / gauss.sum()).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pre, obs):
        batch, seq, height, weight, channel = obs.size()
        pre = (pre * 255.0).int()
        obs= (obs * 255.0).int()
        mse_loss = ((pre-obs)**2).sum()
        return mse_loss/(batch * seq * height * weight * channel)


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, pre, obs):
        batch, seq, height,  weight,  channel = obs.size()
        pre = (pre * 255.0).int()
        obs = (obs * 255.0).int()
        mae_loss = (pre-obs).abs().sum()
        return mae_loss / (batch * seq * height * weight * channel)


class CSI(nn.Module):
    def __init__(self):
        super(CSI, self).__init__()
        self.pixel_to_dBZ =  pixel_to_dBZ

    def forward(self, pre_img, obs_img, dBZ_threshold, eps=1e-6):
        batch, seq, h, w, c = pre_img.shape
        pre_img = self.pixel_to_dBZ(pre_img.detach().cpu().numpy())
        obs_img = self.pixel_to_dBZ(obs_img.detach().cpu().numpy())
        pre_img = np.where(pre_img >= dBZ_threshold, 1, 0)
        obs_img = np.where(obs_img >= dBZ_threshold, 1, 0)

        TP = np.sum((obs_img == 1) & (pre_img== 1))
        FN = np.sum((obs_img == 1) & (pre_img == 0))
        FP = np.sum((obs_img == 0) & (pre_img == 1))
        TN = np.sum((obs_img == 0) & (pre_img == 0))
        csi = TP / (TP + FN + FP + eps)

        return csi

class HSS(nn.Module):
    def __init__(self):
        super(HSS, self).__init__()
        self.pixel_to_dBZ =  pixel_to_dBZ

    def forward(self, pre_img, obs_img, dBZ_threshold, eps=1e-6):
        batch, seq, h, w, c = pre_img.shape
        pre_img = self.pixel_to_dBZ(pre_img.detach().cpu().numpy())
        obs_img = self.pixel_to_dBZ(obs_img.detach().cpu().numpy())
        pre_img = np.where(pre_img >= dBZ_threshold, 1, 0)
        obs_img = np.where(obs_img >= dBZ_threshold, 1, 0)

        TP = np.sum((obs_img == 1) & (pre_img== 1))
        FN = np.sum((obs_img == 1) & (pre_img == 0))
        FP = np.sum((obs_img == 0) & (pre_img == 1))
        TN = np.sum((obs_img == 0) & (pre_img == 0))
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) + eps)

        return hss

class Precision(nn.Module):
    def __init__(self):
        super(Precision, self).__init__()
        self.pixel_to_dBZ =  pixel_to_dBZ

    def forward(self, pre_img, obs_img, dBZ_threshold, eps=1e-6):
        pre_img = self.pixel_to_dBZ(pre_img.detach().cpu().numpy())
        obs_img = self.pixel_to_dBZ(obs_img.detach().cpu().numpy())
        pre_img = np.where(pre_img >= dBZ_threshold, 1, 0)
        obs_img = np.where(obs_img >= dBZ_threshold, 1, 0)

        TP = np.sum((obs_img == 1) & (pre_img == 1))
        FN = np.sum((obs_img == 1) & (pre_img == 0))
        FP = np.sum((obs_img == 0) & (pre_img == 1))
        TN = np.sum((obs_img == 0) & (pre_img == 0))

        prec =  (TP + eps) / (TP + FP + eps)
        return prec

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.pixel_to_dBZ =  pixel_to_dBZ

    def forward(self, pre_img, obs_img, dBZ_threshold, eps=1e-6):
        pre_img = self.pixel_to_dBZ(pre_img.detach().cpu().numpy())
        obs_img = self.pixel_to_dBZ(obs_img.detach().cpu().numpy())
        pre_img = np.where(pre_img >= dBZ_threshold, 1, 0)
        obs_img = np.where(obs_img >= dBZ_threshold, 1, 0)

        TP = np.sum((obs_img == 1) & (pre_img == 1))
        FN = np.sum((obs_img == 1) & (pre_img == 0))
        FP = np.sum((obs_img == 0) & (pre_img == 1))
        TN = np.sum((obs_img == 0) & (pre_img == 0))
        accuracy =   (TP + TN + eps) / (TP + TN + FP + FN + eps)
        return accuracy

class Recall(nn.Module):
    def __init__(self):
        super(Recall, self).__init__()
        self.pixel_to_dBZ =  pixel_to_dBZ

    def forward(self, pre_img, obs_img, dBZ_threshold, eps=1e-6):
        pre_img = self.pixel_to_dBZ(pre_img.detach().cpu().numpy())
        obs_img = self.pixel_to_dBZ(obs_img.detach().cpu().numpy())
        pre_img = np.where(pre_img >= dBZ_threshold, 1, 0)
        obs_img = np.where(obs_img >= dBZ_threshold, 1, 0)

        TP = np.sum((obs_img == 1) & (pre_img == 1))
        FN = np.sum((obs_img == 1) & (pre_img == 0))
        FP = np.sum((obs_img == 0) & (pre_img == 1))
        TN = np.sum((obs_img == 0) & (pre_img == 0))
        recall =   (TP + eps) / (TP + FN + eps)
        return recall

class F1_score(nn.Module):
    def __init__(self):
        super(F1_score, self).__init__()
        self.pixel_to_dBZ =  pixel_to_dBZ

    def forward(self, pre_img, obs_img, dBZ_threshold, eps=1e-6):
        pre_img = self.pixel_to_dBZ(pre_img.detach().cpu().numpy())
        obs_img = self.pixel_to_dBZ(obs_img.detach().cpu().numpy())

        pre_img = np.where(pre_img >= dBZ_threshold, 1, 0)
        obs_img = np.where(obs_img >= dBZ_threshold, 1, 0)

        TP = np.sum((obs_img == 1) & (pre_img == 1))
        FN = np.sum((obs_img == 1) & (pre_img == 0))
        FP = np.sum((obs_img == 0) & (pre_img == 1))
        TN = np.sum((obs_img == 0) & (pre_img == 0))
        precision = (TP + eps) / (TP + FP + eps)
        recall =   (TP + eps) / (TP + FN + eps)
        f1score =   2 * ((precision* recall) / (precision + recall))
        return f1score









