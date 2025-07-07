import pandas as pd
import torch
from lifelines import KaplanMeierFitter
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, w1=1.0, w2=0.1, minp=0.5, eps=1e-8, temperature=1):
        super(ContrastiveLoss, self).__init__()
        self.w1 = w1 # weight for the first term (contrastive loss)
        self.w2 = w2 # weight for the second term (proportion loss)
        self.minp = minp
        self.eps = eps
        self.temperature = temperature # adjusts the scale of the output logits
        # minp is the proportion of patients in cluster 1, if 0, no penalty loss applied
        assert  1 > minp >= 0, "minp should be in the range [0, 1) for the proportion of patients in cluster 1"

    def forward(self, pred, time, event, lambdas, groups, softmax=True):
        """
            :param time: Tensor of shape (N,)
            :param event: Tensor of shape (N,)
            :param lambdas: Tensor of shape (N,)
            :param groups: Tensor of shape (N)
            :param pred: Tensor of shape (N, 2)
            :param softmax: boolean, whether to apply softmax to the predictions
            :return: loss value
        """
        if softmax:
            # Apply softmax to the predictions to get probabilities
            pred = F.softmax(pred / self.temperature, dim=1)
            ind_approx_1 = pred[:, 0]
            ind_approx_2 = pred[:, 1]

        # Calculate the loss
        group_A = (groups == 1).float()  # case subjects
        group_B = (groups == 0).float()  # control subjects

        # t-logrank for Treatment arm A
        g1_num_A = torch.sum(ind_approx_1 * (event - lambdas) * group_A)
        g1_denom_A = torch.sum(ind_approx_1 * lambdas * group_A)

        g2_num_A = torch.sum(ind_approx_2 * (event - lambdas) * group_A)
        g2_denom_A = torch.sum(ind_approx_2 * lambdas * group_A)

        # t-logrank for Treatment arm B
        g1_num_B = torch.sum(ind_approx_1 * (event - lambdas) * group_B)
        g1_denom_B = torch.sum(ind_approx_1 * lambdas * group_B)

        g2_num_B = torch.sum(ind_approx_2 * (event - lambdas) * group_B)
        g2_denom_B = torch.sum(ind_approx_2 * lambdas * group_B)

        test_statistics_A = (g1_num_A ** 2 / (g1_denom_A + 1.0)) + (g2_num_A ** 2 / (g2_denom_A + 1.0))
        test_statistics_B = (g1_num_B ** 2 / (g1_denom_B + 1.0)) + (g2_num_B ** 2 / (g2_denom_B + 1.0))

        # test_statistics_1 = (g1_num_A ** 2 / (g1_denom_A + 1.0)) + (g1_num_B ** 2 / (g1_denom_B + 1.0))
        # test_statistics_2 = (g2_num_A ** 2 / (g2_denom_A + 1.0)) + (g2_num_B ** 2 / (g2_denom_B + 1.0))

        # loss = self.w1 * test_statistics_2 / (test_statistics_1 + test_statistics_2 + self.eps)
        loss = self.w1 * test_statistics_B / (test_statistics_B + test_statistics_A + self.eps)

        if self.minp > 0:
            # proportion loss to balance the number of patients in each cluster
            sum_cluster1 = torch.sum(ind_approx_1)
            sum_cluster2 = torch.sum(ind_approx_2)
            total = sum_cluster1 + sum_cluster2
            # proportion of patients in cluster 1
            pr1 = sum_cluster1 / total
            pr_loss = ((pr1 / self.minp) - 1) ** 2
            loss += pr_loss * self.w2

        return loss


class EntropyLoss(nn.Module):
    def __init__(self, weight=1.0, temperature=1.):
        super(EntropyLoss, self).__init__()
        self.weight = weight
        self.temperature = temperature

    def forward(self, logits: Tensor) -> Tensor:
        """
        :param logits: predicted logits or probability from the model for each cluster, [b, n_clusters]
        :return:
        """
        logits = F.softmax(logits / self.temperature, dim=1)
        entropy = -torch.sum(logits * torch.log(logits + 1e-8), dim=1)
        loss = torch.mean(entropy)
        return loss * self.weight

