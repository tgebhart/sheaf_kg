from typing import Optional
from pykeen.nn.modules import Interaction, parallel_slice_batches
from pykeen.typing import HeadRepresentation, RelationRepresentation, TailRepresentation
import torch
from sheaf_kg import batch_harmonic_extension as bhe

class MultisectionStructuredEmbeddingInteraction(Interaction):

    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    relation_shape = (
        'ed',  # Corresponds to $\mathbf{M}^{head}_j$
        'ed',  # Corresponds to $\mathbf{M}^{tail}_j$
    )
    # entity to be of shape (#cochain_dimension, #num_cochains)
    entity_shape = ('dc',)

    def forward(self, h, r, t):
        # Since the relation_shape is more than length 1, the r value is given as a sequence
        # of the representations defined there. You can use tuple unpacking to get them out
        r_h, r_t = r
        h_proj = r_h @ h
        t_proj = r_t @ t
        closs = -(h_proj - t_proj).norm(p=self.p, dim=-2).sum(dim=-1)
    
        return closs

class BetaeExtensionInteraction(MultisectionStructuredEmbeddingInteraction):

    def __init__(self, p: int = 2):
        super().__init__(p=p)

    def forward(self, h,r,t):
        return super().forward(h,r,t)

    def score_schur_batched(self, edge_index, restriction_maps, 
                            boundary_vertices, interior_vertices,
                            source_vertices, target_vertices, 
                            xS, xT, dv):
        LSchur = bhe.Kron_reduction(edge_index, restriction_maps, boundary_vertices, interior_vertices)
        return bhe.compute_costs(LSchur, source_vertices, target_vertices, xS, xT, dv)
                            
    def score_intersect_batched(self, edge_index, restriction_maps, 
                            source_vertices, target_vertices, 
                            xS, xT, dv):
        L = bhe.Laplacian(edge_index, restriction_maps)
        return bhe.compute_costs(L, source_vertices, target_vertices, xS, xT, dv)
        

