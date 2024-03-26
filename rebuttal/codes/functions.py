import torch, math
import numpy as np
from sklearn.metrics import pairwise_distances
from geom_median.torch import compute_geometric_median
from itertools import combinations

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).T
    dist = xx + yy
    dist.addmm_(1, -2, x, y.T)
    dist[dist < 0] = 0
    dist = dist.sqrt()
    return dist


def FedPG_agg(old_master, world_size, gradient, opts, Batch_size):

    # print("0: ", gradient, gradient[0])
    # raise NotImplementedError

    # flatten the gradient vectors of each worker and put them together, shape [num_worker, -1]
    mu_vec = None
    for idx, item in enumerate(old_master.parameters()):
        # stack gradient[idx] from all worker nodes
        grad_item = []
        for i in range(world_size):
            grad_item.append(gradient[i][idx])
        grad_item = torch.stack(grad_item).view(world_size, -1)
        # print("1: ", grad_item.shape)
        # concat stacked grad vector
        if mu_vec is None:
            mu_vec = grad_item.clone()
        else:
            mu_vec = torch.cat((mu_vec, grad_item.clone()), -1)

    # print("2: ", mu_vec.shape) # (10, 386)
    # raise NotImplementedError
    # calculate the norm distance between each worker's gradient vector, shape [num_worker, num_worker]
    dist = euclidean_dist(mu_vec, mu_vec)

    # calculate C, Variance Bound V, thresold, and alpha
    V = 2 * np.log(2 * opts.num_worker / opts.delta)
    sigma = opts.sigma

    threshold = 2 * sigma * np.sqrt(V / Batch_size)
    alpha = opts.alpha

    # to find MOM: |dist <= threshold| > 0.5 * num_worker
    mu_med_vec = None
    k_prime = (dist <= threshold).sum(-1) > (0.5 * world_size)

    # computes the mom of the gradients, mu_med_vec, and
    # filter the gradients it believes to be Byzantine and store the index of non-Byzantine graidents in Good_set
    if torch.sum(k_prime) > 0:
        mu_mean_vec = torch.mean(mu_vec[k_prime], 0).view(1, -1)
        # select the closet one to the mean from the k_prime list
        mu_med_vec = mu_vec[k_prime][euclidean_dist(mu_mean_vec, mu_vec[k_prime]).argmin()].view(1, -1)
        # applying R1 to filter
        Good_set = euclidean_dist(mu_vec, mu_med_vec) <= 1 * threshold
    else:
        Good_set = k_prime  # skip this step if k_prime is empty (i.e., all False)

    # avoid the scenarios that Good_set is empty or can have |Gt| < (1 − α)K.
    if torch.sum(Good_set) < (1 - alpha) * world_size or torch.sum(Good_set) == 0:

        # re-calculate mom of the gradients
        k_prime = (dist <= 2 * sigma).sum(-1) > (0.5 * world_size)
        if torch.sum(k_prime) > 0:
            mu_mean_vec = torch.mean(mu_vec[k_prime], 0).view(1, -1)
            mu_med_vec = mu_vec[k_prime][euclidean_dist(mu_mean_vec, mu_vec[k_prime]).argmin()].view(1, -1)
            # re-filter with R2
            Good_set = euclidean_dist(mu_vec, mu_med_vec) <= 2 * sigma
        else:
            Good_set = torch.zeros(world_size, 1).to(opts.device).bool()

    # calculate number of good gradients for logging
    N_good = torch.sum(Good_set)

    # aggregate all detected non-Byzantine gradients to get mu
    if N_good > 0:
        mu = []
        for idx, item in enumerate(old_master.parameters()):
            grad_item = []
            for i in range(world_size):
                if Good_set[i]:  # only aggregate non-Byzantine gradients
                    grad_item.append(gradient[i][idx])
            mu.append(torch.stack(grad_item).mean(0))
    else:  # if still all nodes are detected to be Byzantine, check the sigma. If siagma is set properly, this situation will not happen.
        mu = None

    print("Size of the good set: ", N_good)

    # print("3: ", [m.shape for m in mu], mu[-1]) # the mu is of the same shape as gradient and the tensor in mu do not require gradient
    # raise NotImplementedError

    return mu

def SimpleMean(old_master, world_size, gradient, opts):
    Good_set = torch.ones(world_size, 1).to(opts.device).bool()

    mu = []
    for idx, item in enumerate(old_master.parameters()):
        grad_item = []
        for i in range(world_size):
            if Good_set[i]:  # only aggregate non-Byzantine gradients
                grad_item.append(gradient[i][idx])
        mu.append(torch.stack(grad_item).mean(0))

    return mu


def _Krum_update(points, num_corrupt, q):
    points = np.asarray(points)  # (n, d)
    n = points.shape[0]
    # f: expected number of corrupted updates; cap it at n // 2
    num_good = n - num_corrupt - 2  # n - f - 2 # it shows n-f-1 in the appendix
    # multikrum_param = n - num_corrupt  # parameter `m` in the paper # is this q in the appendix? # TODO: make it more flexible
    multikrum_param = q
    sqdist = pairwise_distances(points) ** 2
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = np.sort(sqdist[i])[:num_good + 1].sum()  # exclude k = i

    good_idxs = np.argsort(scores)[:multikrum_param]

    return good_idxs

def _CWTM_update(points, num_points_to_discard):
    if num_points_to_discard == 0:  # no trimming necessary, return simple mean
        return np.average(points, axis=0)
    points = np.asarray(points)  # (n, d)
    aggregated_update = np.zeros_like(points[0])
    # discard at least 1 but do not discard too many
    for i in range(aggregated_update.shape[0]):
        values = np.asarray([p[i] for p in points])
        idxs = np.argsort(values)[num_points_to_discard: -num_points_to_discard]
        aggregated_update[i] = np.average(values[idxs])
    return aggregated_update

def _CWMed_update(points):
    points = np.asarray(points)  # (n, d)
    aggregated_update = np.zeros_like(points[0])
    n = points.shape[0]
    d = points.shape[1]
    if n % 2:
        idxs = np.asarray([(n-1)//2])
    else:
        idxs = np.asarray([n//2-1, n//2])
    # discard at least 1 but do not discard too many
    for i in range(d):
        values = np.asarray([p[i] for p in points])
        values = np.sort(values)
        aggregated_update[i] = np.average(values[idxs])
    return aggregated_update

def _MeaMed_update(points, f):
    points = np.asarray(points)  # (n, d)
    aggregated_update = np.zeros_like(points[0])
    n = points.shape[0]
    d = points.shape[1]
    if n % 2:
        idxs = np.asarray([(n - 1) // 2])
    else:
        idxs = np.asarray([n // 2 - 1, n // 2])

    for i in range(d):
        values = np.asarray([p[i] for p in points])
        values = np.sort(values)
        med = np.average(values[idxs])

        v_dif = [abs(v-med) for v in values]
        idxs = np.argsort(v_dif)[:(n-f)]

        aggregated_update[i] = np.average(values[idxs])

    return aggregated_update

def _coor_update(gradient, opts, is_med, is_mea=False):
    world_size = len(gradient)
    num_com = len(gradient[0])
    f = opts.num_Byzantine

    mu = []
    for idx in range(num_com):
        grad_item = []
        for i in range(world_size):
            grad_item.append(gradient[i][idx])
        grad_item = torch.stack(grad_item).view(world_size, -1)
        mu_vec = grad_item.clone()
        mu_vec = mu_vec.cpu().numpy()  # (n, d)

        # print("1: ", mu_vec.shape)
        if not is_med:
            tmp_mu = _CWTM_update(mu_vec, f)
        else:
            if not is_mea:
                tmp_mu = _CWMed_update(mu_vec)
            else:
                tmp_mu = _MeaMed_update(mu_vec, f)
        # print("2: ", tmp_mu.shape)
        tmp_mu = torch.tensor(tmp_mu, dtype=gradient[0][idx].dtype).view(gradient[0][idx].shape)
        # print("3: ", tmp_mu.shape)
        mu.append(tmp_mu)

    # print("4: ", mu)
    # raise NotImplementedError
    return mu


def _MDA_update(points, cand_sets):
    min_d = math.inf
    min_idx = None
    tot_l = len(cand_sets)

    for i in range(tot_l):
        cur_idx = np.asarray(cand_sets[i])
        cur_points = points[cur_idx]
        # print("1: ", cur_points.shape)
        cur_d = np.max(pairwise_distances(cur_points))
        # print("2: ", pairwise_distances(cur_points), cur_d)
        if cur_d < min_d:
            min_d = cur_d
            min_idx = cur_idx

    return min_idx

def _set_update(gradient, opts, is_krum):

    world_size = len(gradient)
    num_com = len(gradient[0])
    f = opts.num_Byzantine

    if is_krum:
        q = opts.num_worker - f  # q in the appendix, when q == 1 this is Krum, otherwise, MultiKrum # TODO: make it more flexible
    else:
        cand_sets = list(combinations(range(world_size), world_size - f))

    # if we treat gradient from each agent as one vector
    if not opts.compute_per_component:
        mu_vec = None

        for idx in range(num_com):
            # stack gradient[idx] from all worker nodes
            grad_item = []
            for i in range(world_size):
                grad_item.append(gradient[i][idx])
            grad_item = torch.stack(grad_item).view(world_size, -1)
            # print("1: ", grad_item.shape)
            # concat stacked grad vector
            if mu_vec is None:
                mu_vec = grad_item.clone()
            else:
                mu_vec = torch.cat((mu_vec, grad_item.clone()), -1)

        mu_vec = mu_vec.cpu().numpy()  # (n, d)

        # print("1: ", mu_vec.shape)
        if is_krum:
            Good_set = _Krum_update(mu_vec, f, q)
        else:
            Good_set = _MDA_update(mu_vec, cand_sets)

        mu = []
        for idx in range(num_com):
            grad_item = []
            for i in Good_set:
                grad_item.append(gradient[i][idx])
            mu.append(torch.stack(grad_item).mean(0))

        # print("2: ", mu, [m.shape for m in mu])
        # raise NotImplementedError

    else:
        mu = []
        for idx in range(num_com):
            grad_item = []
            for i in range(world_size):
                grad_item.append(gradient[i][idx])
            grad_item = torch.stack(grad_item).view(world_size, -1)
            mu_vec = grad_item.clone()
            mu_vec = mu_vec.cpu().numpy()  # (n, d)
            # print("1: ", mu_vec.shape)

            if is_krum:
                Good_set = _Krum_update(mu_vec, f, q)
            else:
                Good_set = _MDA_update(mu_vec, cand_sets)

            grad_item = []
            for i in Good_set:
                grad_item.append(gradient[i][idx])
            mu.append(torch.stack(grad_item).mean(0))

    # print("2: ", mu, [m.shape for m in mu])
    # raise NotImplementedError

    return mu


def MDA(gradient, opts):
    return _set_update(gradient, opts, is_krum=False)

def Krum(gradient, opts):
    return _set_update(gradient, opts, is_krum=True)

def CWTM(gradient, opts):
    return _coor_update(gradient, opts, is_med=False)

def CWMed(gradient, opts):
    return _coor_update(gradient, opts, is_med=True)

def MeaMed(gradient, opts):
    return _coor_update(gradient, opts, is_med=True, is_mea=True)

def GM(gradients, opts):
    # gradients is a list of 'x', and 'x' is a list of tensors, and each tensor is a gradient for a parameter matrix
    out = compute_geometric_median(gradients, weights=None, per_component=opts.compute_per_component) # per_component=True, or not

    # print("1: ", out.median, out)
    # raise NotImplementedError

    return out.median