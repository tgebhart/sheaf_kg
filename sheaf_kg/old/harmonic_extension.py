import numpy as np
from scipy import linalg

# edge_index is (2,ne)
# restriction maps is (ne,2,de,dv)

def coboundary(edge_index,restriction_maps):
    ne = edge_index.shape[1]
    nv = np.max(edge_index) + 1 #assume there are vertices indexed 0...max
    de = restriction_maps.shape[2]
    dv = restriction_maps.shape[3]
    d = np.zeros((ne*de,nv*dv))
    for e in range(ne):
        h = edge_index[0,e]
        t = edge_index[1,e]
        d[e*de:(e+1)*de,h*dv:(h+1)*dv] = restriction_maps[e,0,:,:]
        d[e*de:(e+1)*de,t*dv:(t+1)*dv] = -restriction_maps[e,1,:,:]
    return d

def Laplacian(edge_index,restriction_maps):
    d = coboundary(edge_index,restriction_maps)
    return d.T @ d

def get_matrix_indices(vertices,dv):
    Midx = np.zeros(len(vertices)*dv,dtype=np.int)
    for (i,v) in enumerate(vertices):
        Midx[i*dv:(i+1)*dv] = range(v*dv,(v+1)*dv)
    return Midx

def harmonic_extension(edge_index,restriction_maps,boundary_vertices,interior_vertices,xB):
    L = Laplacian(edge_index,restriction_maps)
    dv = restriction_maps.shape[3]
    Bidx = get_matrix_indices(boundary_vertices,dv)
    Uidx = get_matrix_indices(interior_vertices,dv)
    LBB = L[np.ix_(Bidx,Bidx)]
    LUB = L[np.ix_(Uidx,Bidx)]
    LUU = L[np.ix_(Uidx,Uidx)]

    xU = -linalg.pinv(LUU) @ LUB @ xB

def Kron_reduction(edge_index,restriction_maps,boundary_vertices,interior_vertices):
    L = Laplacian(edge_index,restriction_maps)
    dv = restriction_maps.shape[3]
    Bidx = get_matrix_indices(boundary_vertices,dv)
    Uidx = get_matrix_indices(interior_vertices,dv)

    LBB = L[np.ix_(Bidx,Bidx)]
    LUB = L[np.ix_(Uidx,Bidx)]
    LUU = L[np.ix_(Uidx,Uidx)]

    schur = LBB - LUB.T @ linalg.pinv(LUU) @ LUB
    return schur

def compute_costs(L,source_vertices,target_vertices,xS,xT,dv):
    # xS should just be a vector with the embeddings of the known entities, xT a (|S|*dv, num_entities)
    # matrix whose slices contain the embeddings of the entities to be checked.
    # returns a vector with entries Q([xS; xT[:,i]]), where Q(x) is the quadratic form x^TLx
    # running time should be linear in the number of entities.
    Sidx = get_matrix_indices(source_vertices,dv)
    Tidx = get_matrix_indices(target_vertices,dv)

    LSS = L[np.ix_(Sidx,Sidx)]
    LST = L[np.ix_(Sidx,Tidx)]
    LTT = L[np.ix_(Tidx,Tidx)]

    const = xS.T @ LSS @ xS
    lin = xS.T @ LST @ xT
    quad = np.sum(xT * (LTT @ xT), axis=0)

    return const + lin + quad
