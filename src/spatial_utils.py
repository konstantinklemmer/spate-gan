import torch
import scipy.sparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from libpysal.weights import lat2W

#Convert sparse scipy matrix to torch sparse tensor
def crs_to_torch_sparse(x):
    #
    # Input:
    # x = crs matrix (scipy sparse matrix)
    # Output:
    # w = weight matrix as toch sparse tensor
    x = x.tocoo()
    values = x.data
    indices = np.vstack((x.row, x.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = x.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

# Create spatial weight matrix
def make_sparse_weight_matrix(h, w, rook=False):
    #
    # Input:
    # h = height
    # w = width
    # rook = use rook weights or not
    # Output:
    # w = weight matrix as toch sparse tensor
    w = lat2W(h, w, rook=rook)
    return crs_to_torch_sparse(w.sparse)

# Compute temporal distance weight tensor
def temporal_weights(n,b):
    #
    # Input:
    # n = number of time steps
    # b = parameter governing exponential weight decay
    # Output:
    # weights = temporal weights
    weights = torch.exp(-torch.arange(1,n).flip(0) / b).view(1,1,-1)
    return weights

# Space-time expectations (assuming temporal order to suit sequentiality constraints)
def st_ex(x, weights):
    #
    # Input:
    # x = input video of shape [height, width, n_frames]
    # weights = tensor of distance weights of shape [1, 1, time_steps]
    # (Can be computed via temporal_weights())
    # Output:
    # exp_val = expected values assuming space-time independence and sequential calculation;
    # shape [height, width, n_frames-1]
    h, w, n = x.shape
    exp_val = [(weights[:, :, -t:] * x[:, :, :t]).sum(dim=2).reshape(-1) * x[:, :, t].reshape(-1).sum() / (weights[:, :, -t:] * x[:, :, :t]).reshape(-1).sum() for t in range(1, n)]
    exp_val = torch.stack(exp_val).permute(1, 0).reshape(h, w, n - 1)
    return exp_val

# Space-time expectations; as proposed by Kulldorff, 2005 (assuming knowledge of the whole time series; no temporal weights)
def st_ex_kulldorff(x):
    #
    # Input:
    # x = input video of shape [height, width, n_frames]
    # Output:
    # exp_val = expected values assuming space-time independence and sequential calculation;
    # shape [height, width, n_frames-1]
    h, w, n = x.shape
    s_ex = torch.stack([x[:, :, t].reshape(-1).sum() for t in range(0, n)])
    t_ex = x.sum(dim=2)
    exp_val = torch.einsum('ab,c->abc', (t_ex, s_ex)) / x.reshape(-1).sum()
    return exp_val

# Space-time expectations as proposed by Kulldorff
# (assuming knowledge of the whole time series; including temporal weights)
def st_ex_kulldorff_weighted(x, weights):
    #
    # Input:
    # x = input video of shape [height, width, n_frames]
    # weights = tensor of distance weights of shape [1, 1, time_steps]
    # (Can be computed via temporal_weights())
    # Output:
    # exp_val = expected values assuming space-time independence and sequential calculation;
    # shape [height, width, n_frames-1]
    h, w, n = x.shape
    exp_val = torch.stack([x[:, :, t].reshape(-1).sum() * (x * weights[t,...].reshape(1, 1, -1)).sum(dim=2) / (x * weights[t,...].reshape(1,1,-1)).reshape(-1).sum() for t in range(0,n)]).permute(1,2,0)
    return exp_val

# Local Moran's I with custom means
def mi_mean(x, x_mean, w_sparse):
    #
    # Input:
    # x = input data tensor (flattened or image)
    # x_mean = input tensor of same (flattened) shape as x
    # w_sparse = spatial weight matrix; torch sparse tensor
    # Output:
    # mi = output data - local Moran's I
    #
    x = x.reshape(-1)
    n = len(x)
    n_1 = n - 1
    z = x - x_mean
    sx = x.std()
    z /= sx
    den = (z * z).sum()
    zl = torch.sparse.mm(w_sparse, z.reshape(-1, 1)).reshape(-1)
    mi = n_1 * z * zl / den
    return mi


# Local Moran's I
def mi(x, w_sparse):
    #
    # Input:
    # x = input data tensor (flattened or image)
    # w_sparse = spatial weight matrix; torch sparse tensor
    # Output:
    # mi = output data - local Moran's I
    #
    x = x.reshape(-1)
    n = len(x)
    n_1 = n - 1
    z = x - x.mean()
    sx = x.std()
    z /= sx
    den = (z * z).sum()
    zl = torch.sparse.mm(w_sparse, z.reshape(-1, 1)).reshape(-1)
    mi = n_1 * z * zl / den
    return mi

# Local Moran's I for a video (time-series of images)
def vid_mi(x, w_sparse):
    #
    # Input:
    # x = input video of shape [height, width, n_frames]
    # w_sparse = spatial weight matrix; torch sparse tensor
    # Output:
    # mis = output data - local Moran's Is
    #
    h, w, n = x.shape
    mis = torch.stack([mi(x[:, :, i].reshape(-1), w_sparse).reshape(h, w) for i in range(n)])
    return mis

# Make Local Moran's I for a batch of videos
def make_mis(x, w_sparse):
    #
    # Input:
    # x = input video batch of shape [batch_size, time_steps, n_channel, height, width]
    # w_sparse = spatial weight matrix; torch sparse tensor
    # Output:
    # mis = output data - local Moran's I
    #
    n, t, nc, h, w = x.shape
    mis = torch.stack([vid_mi(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), w_sparse) for j in range(nc) for i in range(n)]).reshape(n, t, nc, h, w)
    mis = torch.stack([(mis[i, :, j, :, :] - torch.min(mis[i, :, j, :, :])) / (torch.max(mis[i, :, j, :, :]) - torch.min(mis[i, :, j, :, :])) for j in range(nc) for i in range(n)]).reshape(n, t, nc, h, w)
    return mis

# SPATE: Local Moran's I for video data, using space-time expectations
def spate(x, w_sparse, b, method="skw"):
    #
    # Input:
    # x = input video of shape [height, width, n_frames]
    # w_sparse = spatial weight matrix; scipy sparse matrix
    # b = tensor of distance weights of shape [1, 1, time_steps]
    # (Can be computed via temporal_weights())
    # method = method to use for computing space-time expectations; default 'skw'
    # (Options are sequential Kulldorff-weighted ('skw'), Kulldorff ('k'), Kulldorff-weighted ('kw'))
    # Output:
    # spates = output data - SPATE
    #
    h, w, n = x.shape
    if method == "k":
        x_means = st_ex_kulldorff(x)
    elif method == "kw":
        x_means = st_ex_kulldorff_weighted(x, b)
    else:
        x_means = st_ex(x, b)
    if method=="skw":
        spates = torch.stack([mi_mean(x[:, :, i + 1].reshape(-1), x_means[:, :, i].reshape(-1), w_sparse).reshape(h, w) for i in range(n - 1)])
    else:
        spates = torch.stack([mi_mean(x[:, :, i].reshape(-1), x_means[:, :, i].reshape(-1), w_sparse).reshape(h, w) for i in range(n)])
    return spates.permute(1, 2, 0)

# Make SPATEs for a batch of videos
def make_spates(x, w_sparse, b, method="skw"):
    #
    # Input:
    # x = input video batch of shape [batch_size, time_steps, n_channel, height, width]
    # w_sparse = spatial weight matrix; torch sparse tensor
    # b = tensor of distance weights of shape [1, 1, time_steps]
    # (Can be computed via temporal_weights())
    # method = method to use for computing space-time expectations; default 'skw'
    # (Options are sequential Kulldorff-weighted ('skw'), Kulldorff ('k'), Kulldorff-weighted ('kw'))
    # Output:
    # spates = output data - SPATE
    #
    n, t, nc, h, w = x.shape
    if method=="skw":
        spates = torch.stack([spate(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), w_sparse, b, method).permute(2, 0, 1) for j in range(nc) for i in range(n)]).reshape(n, t - 1, nc, h, w)
        spates = torch.stack([(spates[i, :, j, :, :] - torch.min(spates[i, :, j, :, :])) / (torch.max(spates[i, :, j, :, :]) - torch.min(spates[i, :, j, :, :])) for j in range(nc) for i in range(n)]).reshape(n, t - 1, nc, h, w)
        spates = F.pad(spates, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        spates = torch.roll(spates, 1, 1)
    else:
        spates = torch.stack([spate(x[i, :, j, :, :].reshape(t, h, w).permute(1, 2, 0), w_sparse, b, method).permute(2, 0, 1) for j in range(nc) for i in range(n)]).reshape(n, t, nc, h, w)
        spates = torch.stack([(spates[i, :, j, :, :] - torch.min(spates[i, :, j, :, :])) / (torch.max(spates[i, :, j, :, :]) - torch.min(spates[i, :, j, :, :])) for j in range(nc) for i in range(n)]).reshape(n, t, nc, h, w)
    return spates

# Convert point-process intensities to point coordinates by uniformly distributing points in grid cells
def intensity_to_points(x, beta=50,set_noise=True,theta=0.5):
  #
  # Input:
  # x = input video of spatio-temporal point-process intensities of shape [n_frames, no_channels, height, width]
  # beta = controls the number of points to generate (multiplies with the intensity values x); default = 50
  # set_noise = should random noise be added to the generated coordinates; default = True
  # theta = controls the amount of noise to be added (multiplies with the generated Gaussian random noise); default = 0.5
  # Output:
  # t_points = spatio-temporal point pattern of shape [n, 3] with spatio-temporal coordinates x,y,z
  t, nc, h, w = x.shape
  assert nc == 1, "Only univariate point-process intensities supported"
  x_grid = torch.arange(0,h)
  y_grid = torch.arange(0,w)
  indices = torch.tensor(np.array(list(product(x_grid, y_grid)))).flip(dims=[0,1])
  t_points = []
  for i in range(t):
    d_step = x[i,...].reshape(-1) 
    m = torch.div(d_step * beta, 1, rounding_mode="floor")
    points = [i for item, count in zip(indices, m) for i in [item] * count.int()]
    points = torch.cat(points).reshape(-1,2)
    ts = torch.tensor([i] * points.shape[0]).reshape(-1,1)
    if set_noise:
      noise = torch.randn(points.shape) * theta
      points = points + noise
    points = torch.cat([points,ts],dim=1)
    t_points.append(points)
  return torch.cat(t_points)