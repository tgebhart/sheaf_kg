import torch
import numpy as np
# edge_index is (2,ne)
# restriction maps is (nbatch,ne,2,de,dv)

def coboundary(edge_index,restriction_maps):
    device = restriction_maps.device
    nb = restriction_maps.shape[0]
    ne = edge_index.shape[-1]
    nv = torch.max(edge_index) + 1 #assume there are vertices indexed 0...max
    de = restriction_maps.shape[-2]
    dv = restriction_maps.shape[-1]
    d = torch.zeros((nb,ne*de,nv*dv), device=device)
    for e in range(ne):
        h = edge_index[0,e]
        t = edge_index[1,e]
        d[:,e*de:(e+1)*de,h*dv:(h+1)*dv] = restriction_maps[:,e,0,:,:]
        d[:,e*de:(e+1)*de,t*dv:(t+1)*dv] = -restriction_maps[:,e,1,:,:]
    return d

def Laplacian(edge_index,restriction_maps):
    d = coboundary(edge_index,restriction_maps)
    return torch.matmul(torch.transpose(d,1,2), d)

def get_matrix_indices(vertices,dv):
    Midx = torch.zeros(vertices.shape[0]*dv,dtype=torch.int64)
    for (i,v) in enumerate(vertices):
        Midx[i*dv:(i+1)*dv] = torch.arange(v*dv,(v+1)*dv)
    return Midx

def harmonic_extension(edge_index,restriction_maps,boundary_vertices,interior_vertices,xB):
    L = Laplacian(edge_index,restriction_maps)
    dv = restriction_maps.shape[-1]
    nbatch = L.shape[0]

    Bidx = get_matrix_indices(boundary_vertices,dv)
    Uidx = get_matrix_indices(interior_vertices,dv)
    batch_idx = np.arange(nbatch)

    LBB = L[np.ix_(batch_idx,Bidx,Bidx)]
    LUB = L[np.ix_(batch_idx,Uidx,Bidx)]
    LUU = L[np.ix_(batch_idx,Uidx,Uidx)]

    xU = -torch.linalg.pinv(LUU, rcond=1e-15, hermitian=True) @ LUB @ xB

def Kron_reduction(edge_index,restriction_maps,boundary_vertices,interior_vertices):
    L = Laplacian(edge_index,restriction_maps)
    dv = restriction_maps.shape[-1]
    nbatch = L.shape[0]

    Bidx = get_matrix_indices(boundary_vertices,dv)
    Uidx = get_matrix_indices(interior_vertices,dv)
    batch_idx = np.arange(nbatch)

    LBB = L[np.ix_(batch_idx,Bidx,Bidx)]
    LUB = L[np.ix_(batch_idx,Uidx,Bidx)]
    LUU = L[np.ix_(batch_idx,Uidx,Uidx)]

    # lstsq recommended by pytorch docs https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html
    schur = LBB - torch.transpose(LUB, 1, 2) @ torch.linalg.lstsq(LUU, LUB).solution

    return schur

def Kron_reduction_translational(edge_index,restriction_maps,boundary_vertices,interior_vertices):
    L = Laplacian(edge_index, restriction_maps)
    d = coboundary(edge_index, restriction_maps)
    dv = restriction_maps.shape[-1]
    nbatch = L.shape[0]

    Bidx = get_matrix_indices(boundary_vertices,dv)
    Uidx = get_matrix_indices(interior_vertices,dv)
    batch_idx = np.arange(nbatch)

    LBB = L[np.ix_(batch_idx,Bidx,Bidx)]
    LUB = L[np.ix_(batch_idx,Uidx,Bidx)]
    LUU = L[np.ix_(batch_idx,Uidx,Uidx)]

    didx = np.arange(d.shape[1])
    dU = d[np.ix_(batch_idx, didx, Uidx)]
    dB = d[np.ix_(batch_idx, didx, Bidx)]

    invLUB = torch.linalg.lstsq(LUU, LUB).solution

    schur = LBB - torch.transpose(LUB, 1, 2) @ invLUB
    affine = -dU @ invLUB + dB
    return schur, affine

def compute_costs(L,source_vertices,target_vertices,xS,xT,dv):
    """xS should be a matrix of (n_batch, num_source * dv). xT should be
    a matrix of (num_targets, dv).
    """
    nbatch = L.shape[0]

    Sidx = get_matrix_indices(source_vertices,dv)
    Tidx = get_matrix_indices(target_vertices,dv)
    batch_idx = np.arange(nbatch)

    LSS = L[np.ix_(batch_idx,Sidx,Sidx)]
    LST = L[np.ix_(batch_idx,Sidx,Tidx)]
    LTT = L[np.ix_(batch_idx,Tidx,Tidx)]

    # rewrite diagonal -- unnecessary computation
    const = torch.sum(xS * (LSS @ xS), axis=1)
    xT = xT.unsqueeze(1)
    lin = 2 * torch.sum(xS * (LST @ xT), axis=2)
    quad = torch.sum(xT * (LTT @ xT), axis=2)
    return torch.transpose((const[None, :, :] + lin + quad), 0, 1)

def compute_costs_translational(L,affine,source_vertices,target_vertices,xS,xT,b,dv):
    """xS should be a matrix of (n_batch, num_source * dv). xT should either be
    a matrix of (n_batch, S*dv) or a matrix of (num_targets, dv).
    """
    nbatch = L.shape[0]

    Sidx = get_matrix_indices(source_vertices,dv)
    Tidx = get_matrix_indices(target_vertices,dv)
    batch_idx = np.arange(nbatch)

    LSS = L[np.ix_(batch_idx,Sidx,Sidx)]
    LST = L[np.ix_(batch_idx,Sidx,Tidx)]
    LTT = L[np.ix_(batch_idx,Tidx,Tidx)]

    affSS = affine[np.ix_(batch_idx, np.arange(affine.shape[1]), Sidx)]
    affTT = affine[np.ix_(batch_idx, np.arange(affine.shape[1]), Tidx)]

    const = torch.sum(xS * (LSS @ xS), axis=1)
    xT = xT.unsqueeze(1)
    lin = 2 * torch.sum(xS * (LST @ xT), axis=2)
    quad = torch.sum(xT * (LTT @ xT), axis=2)
    affine = 2*torch.sum(b * ((affSS @ xS) + (affTT @ xT)), axis=2)
    return torch.transpose((const[None, :, :] + lin + quad + affine), 0, 1)