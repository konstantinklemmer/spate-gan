import os
import json
import argparse
import time
from datetime import datetime

import cv2
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

def cost_matrix(x, y, p=2, scale=False):
    '''
    L2 distance between vectors, using expanding and hence is more memory intensive
    :param x: x is tensor of shape [batch_size, time steps, features]
    :param y: y is tensor of shape [batch_size, time steps, features]
    :param p: power
    :return: cost matrix: a matrix of size [batch_size, batch_size] where
    '''
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    time_steps = x.shape[1]
    b = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    c = torch.sum(b, -1)
    if scale:
        c /= time_steps
    return c


def modified_cost(x, y, h, M, scale=False):
    '''
    :param x: a tensor of shape [batch_size, time steps, features]
    :param y: a tensor of shape [batch_size, time steps, features]
    :param h: a tensor of shape [batch size, time steps, J]
    :param M: a tensor of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :return: L1 cost matrix plus h, M modification:
    a matrix of size [batch_size, batch_size] where
    c_hM_{ij} = c_hM(x^i, y^j) = L2_cost + \sum_{t=1}^{T-1}h_t\Delta_{t+1}M
    ====> NOTE: T-1 here, T = # of time steps
    '''
    # compute sum_{t=1}^{T-1} h[t]*(M[t+1]-M[t])
    DeltaMt = M[:, 1:, :] - M[:, :-1, :]
    ht = h[:, :-1, :]
    time_steps = ht.shape[1]
    sum_over_j = torch.sum(ht[:, None, :, :] * DeltaMt[None, :, :, :], -1)
    C_hM = torch.sum(sum_over_j, -1)
    if scale:
        C_hM /= time_steps

    # Compute L2 cost $\sum_t^T |x^i_t - y^j_t|^2$
    cost_xy = cost_matrix(x, y, scale=scale)

    return cost_xy + C_hM


def compute_sinkhorn(x, y, h, M, epsilon=0.1, niter=10, scale=False, benchmark=False):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """
    n = x.shape[0]

    # The Sinkhorn algorithm takes as input three variables :
    if benchmark:
        C = cost_matrix(x, y, scale=scale)
    else:
        C = modified_cost(x, y, h, M, scale=scale) # shape: [batch_size, batch_size]


    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    mu = 1. / n * torch.ones(n, requires_grad=False, device=x.device)
    nu = 1. / n * torch.ones(n, requires_grad=False, device=x.device)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-4)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.logsumexp(A, dim=-1, keepdim=True)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).item():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost


def scale_invariante_martingale_regularization(M, reg_lam, scale=False):
    '''
    Compute the regularization for the martingale condition (i.e. p_M).
    :param M: a tensor of shape (batch_size, sequence length), the output of an RNN applied to X
    :param reg_lam: scale parameter for first term in pM
    :return: A rank 0 tensors (i.e. scalers)
    This tensor represents the martingale penalization term denoted $p_M$
    '''
    m, t, j = M.shape
    # m = torch.tensor(m).type(torch.FloatTensor)
    # t = torch.tensor(m).type(torch.FloatTensor)
    # compute delta M matrix N
    N = M[:, 1:, :] - M[:, :-1, :]
    N_std = N / (torch.std(M, (0, 1)) + 1e-06)

    # Compute \sum_i^m(\delta M)
    sum_m_std = torch.sum(N_std, 0) / m
    # Compute martingale penalty: P_M1 =  \sum_i^T(|\sum_i^m(\delta M)|) * scaling_coef
    sum_across_paths = torch.sum(torch.abs(sum_m_std))
    if scale:
        sum_across_paths /= t

    # the total pM term
    pm = reg_lam * sum_across_paths
    return pm


def compute_mixed_sinkhorn_loss(f_real, f_fake, m_real, m_fake, h_fake, sinkhorn_eps, sinkhorn_l,
                                f_real_p, f_fake_p, m_real_p, h_real_p, h_fake_p, scale=False):
    '''
    :param x and x'(f_real, f_real_p): real data of shape [batch size, time steps, features]
    :param y and y'(f_fake, f_fake_p): fake data of shape [batch size, time steps, features]
    :param h and h'(h_real, h_fake): h(y) of shape [batch size, time steps, J]
    :param m and m'(m_real and m_fake): M(x) of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and actual number of sinkhorn iterations for monitoring the training process)
    '''
    f_real = f_real.reshape(f_real.shape[0], f_real.shape[1], -1)
    f_fake = f_fake.reshape(f_fake.shape[0], f_fake.shape[1], -1)
    f_real_p = f_real_p.reshape(f_real_p.shape[0], f_real_p.shape[1], -1)
    f_fake_p = f_fake_p.reshape(f_fake_p.shape[0], f_fake_p.shape[1], -1)
    loss_xy = compute_sinkhorn(f_real, f_fake, h_fake, m_real, sinkhorn_eps, sinkhorn_l, scale=scale)
    loss_xyp = compute_sinkhorn(f_real_p, f_fake_p, h_fake_p, m_real_p, sinkhorn_eps, sinkhorn_l, scale=scale)
    loss_xx = compute_sinkhorn(f_real, f_real_p, h_real_p, m_real, sinkhorn_eps, sinkhorn_l, scale=scale)
    loss_yy = compute_sinkhorn(f_fake, f_fake_p, h_fake_p, m_fake, sinkhorn_eps, sinkhorn_l, scale=scale)

    loss = loss_xy + loss_xyp - loss_xx - loss_yy
    return loss


def compute_classic_sinkhorn_loss(f_real, f_fake, m_real, m_fake, h_fake, h_real, sinkhorn_eps,
                                  sinkhorn_l, scale=False):
    '''
    :param x and x'(f_real, f_real_p): real data of shape [batch size, time steps, features]
    :param y and y'(f_fake, f_fake_p): fake data of shape [batch size, time steps, features]
    :param h and h'(h_real, h_fake): h(y) of shape [batch size, time steps, J]
    :param m and m'(m_real and m_fake): M(x) of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and actual number of sinkhorn iterations for monitoring the training process)
    '''
    f_real = f_real.reshape(f_real.shape[0], f_real.shape[1], -1)
    f_fake = f_fake.reshape(f_fake.shape[0], f_fake.shape[1], -1)
    loss_xy = compute_sinkhorn(f_real, f_fake, h_fake, m_real, sinkhorn_eps, sinkhorn_l, scale=scale)
    loss_xx = compute_sinkhorn(f_real, f_real, h_real, m_real, sinkhorn_eps, sinkhorn_l, scale=scale)
    loss_yy = compute_sinkhorn(f_fake, f_fake, h_fake, m_fake, sinkhorn_eps, sinkhorn_l, scale=scale)

    loss = 2.0 * loss_xy - loss_xx - loss_yy
    return loss


def original_sinkhorn_loss(x, y, sinkhorn_eps, sinkhorn_l, scale=False):
    '''
    :param x: real data of shape [batch size, time steps, features]
    :param y: fake data of shape [batch size, time steps, features]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and actual number of sinkhorn iterations for monitoring the training process)
    '''
    loss_xy = compute_sinkhorn(x, y, sinkhorn_eps, sinkhorn_l, scale=scale, benchmark=True)
    loss_xx = compute_sinkhorn(x, x, sinkhorn_eps, sinkhorn_l, scale=scale, benchmark=True)
    loss_yy = compute_sinkhorn(y, y, sinkhorn_eps, sinkhorn_l, scale=scale, benchmark=True)

    loss = 2.0 * loss_xy - loss_xx - loss_yy

    return loss