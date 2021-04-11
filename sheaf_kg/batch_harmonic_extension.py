import torch
import numpy as np
# edge_index is (nbatch,2,ne)
# restriction maps is (nbatch,ne,2,de,dv)

def coboundary(edge_index,restriction_maps):
    nb = edge_index.shape[0]
    ne = edge_index.shape[2]
    nv = torch.max(edge_index) + 1 #assume there are vertices indexed 0...max
    de = restriction_maps.shape[-2]
    dv = restriction_maps.shape[-1]
    d = torch.zeros((nb,ne*de,nv*dv))
    for e in range(ne):
        h = edge_index[0,0,e]
        t = edge_index[0,1,e]
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

    LBB = L[np.ix_(np.arange(nbatch),Bidx,Bidx)]
    LUB = L[np.ix_(np.arange(nbatch),Uidx,Bidx)]
    LUU = L[np.ix_(np.arange(nbatch),Uidx,Uidx)]

    xU = -torch.linalg.pinv(LUU, rcond=1e-10, hermitian=True) @ LUB @ xB

def Kron_reduction(edge_index,restriction_maps,boundary_vertices,interior_vertices):
    L = Laplacian(edge_index,restriction_maps)
    dv = restriction_maps.shape[-1]
    nbatch = L.shape[0]

    Bidx = get_matrix_indices(boundary_vertices[0],dv)
    Uidx = get_matrix_indices(interior_vertices[0],dv)

    LBB = L[np.ix_(np.arange(nbatch),Bidx,Bidx)]
    LUB = L[np.ix_(np.arange(nbatch),Uidx,Bidx)]
    LUU = L[np.ix_(np.arange(nbatch),Uidx,Uidx)]

    schur = LBB - torch.transpose(LUB, 1, 2) @ torch.linalg.pinv(LUU, rcond=1e-10, hermitian=True) @ LUB
    return schur

def compute_costs(L,source_vertices,target_vertices,xS,xT,dv):
    """xS should be a matrix of (n_batch, num_source * dv). xT should either be
    a matrix of (n_batch, S*dv) or a matrix of (num_targets, dv).
    """
    nbatch = L.shape[0]

    Sidx = get_matrix_indices(source_vertices[0],dv)
    Tidx = get_matrix_indices(target_vertices[0],dv)

    LSS = L[np.ix_(np.arange(nbatch),Sidx,Sidx)]
    LST = L[np.ix_(np.arange(nbatch),Sidx,Tidx)]
    LTT = L[np.ix_(np.arange(nbatch),Tidx,Tidx)]

    if xS.shape[0] != xT.shape[0]:
        const = torch.diagonal(torch.transpose(xS, -2, -1) @ LSS @ xS, dim1=-2, dim2=-1)
        xT = xT.unsqueeze(1)
        lin = torch.diagonal(torch.transpose(xS, -2, -1) @ LST @ xT, dim1=-2, dim2=-1)
        inner = (LTT @ xT)
        quad = torch.sum(xT * (LTT @ xT), axis=2)
        return torch.transpose((const[None, :, :] + lin + quad), 0, 1)
    else:
        const = torch.diagonal(torch.transpose(xS, -2, -1) @ LSS @ xS, dim1=-2, dim2=-1)
        lin = torch.diagonal(torch.transpose(xS, -2, -1) @ LST @ xT, dim1=-2, dim2=-1)
        quad = torch.sum(xT * (LTT @ xT), axis=1)
        return const + lin + quad
