from typing import Optional
from pykeen.nn.modules import Interaction, parallel_slice_batches
from pykeen.typing import HeadRepresentation, RelationRepresentation, TailRepresentation
import torch
from sheaf_kg import tensor_harmonic_extension as the
import networkx as nx

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

class ExtensionInteraction(MultisectionStructuredEmbeddingInteraction):

    def __init__(self, p: int = 2, training_mask_pct: float = 0):
        super().__init__(p=p)
        if training_mask_pct > 1 or training_mask_pct < 0:
            raise ValueError(f'training mask percentage must be between 0 and 1, got {training_mask_pct}')
        self.training_mask_pct = training_mask_pct

    # def _forward_harmonic_extension(self, boundary, r, interior, to_score):
    #     # Since the relation_shape is more than length 1, the r value is given as a sequence
    #     # of the representations defined there. You can use tuple unpacking to get them out
    #     r_h, r_t = r

    #     # get unique indices across batch dimension
    #     # a little hacky, but it doesn't appear possible to get indices from `self`
    #     _, b_idxs = torch.unique(boundary, dim=0, return_inverse=True, sorted=False)
    #     _, i_idxs = torch.unique(interior, dim=0, return_inverse=True, sorted=False)

    #     edge_index = torch.cat([b_idxs.unsqueeze(0), i_idxs.unsqueeze(0)], dim=0)
    #     restriction_maps = torch.cat([r_h, r_t], dim=1)

    #     Lschur = the.Kron_reduction(edge_index, restriction_maps, b_idxs, i_idxs)
    #     costs = the.compute_costs(Lschur, b_idxs, i_idxs, boundary.reshape(Lschur.shape[0], -1), to_score, boundary.shape[-2] )
    #     costs = costs.repeat((b_idxs.shape[0], 1))
    #     return costs

    def _forward_harmonic_extension(self, sources, r, targets, to_score):
        # broadly, we wish to compute scores via harmonic extension for each 
        # triplet in the incoming batch. for each triplet i, our approach will 
        # be to treat the full batch as a subgraph and take the schur complement 
        # of all of the batch entities that are not the head or tail entities of i. 

        # we could alter this to take multiple source vertices.
        # the sampling for such an approach seems arbitrary without a 
        # particular complex query structure in mind.

        # unpack restriction maps
        r_h, r_t = r

        # get unique indices across batch dimension
        # a little hacky, but it doesn't appear possible to get indices from `self`
        _, s_idxs = torch.unique(sources, dim=0, return_inverse=True, sorted=False)
        _, t_idxs = torch.unique(targets, dim=0, return_inverse=True, sorted=False)

        # edge_index is (2,ne)
        edge_index = torch.cat([s_idxs.unsqueeze(0), t_idxs.unsqueeze(0)], dim=0)
        # restriction maps is (ne,2,de,dv)
        restriction_maps = torch.cat([r_h, r_t], dim=1)
        
        # store scores for each triplet
        scores = torch.empty((sources.shape[0], to_score.shape[1]))

        for i in range(sources.shape[0]):
            # source vertex representation
            sv = sources[i]
            # boundary indices
            b_idxs = edge_index[:,i]
            # interior indices which are not equal to either of the boundary indices
            i_idxs = edge_index[:,(edge_index[0,:] != b_idxs[0]) & (edge_index[1,:] != b_idxs[1])]
            # restriction maps of the above relations 
            i_rmaps = restriction_maps[(edge_index[0,:] != b_idxs[0]) & (edge_index[1,:] != b_idxs[1]),:,:,:]

            # add the head and tail entities of batch triplet i
            filtered_edge_index = torch.cat([b_idxs.unsqueeze(1), i_idxs], dim=1)
            # add the restriction maps for triplet i
            filtered_restriction_maps = torch.cat([restriction_maps[i].unsqueeze(0), i_rmaps], dim=0)

            # take the schur complement over the interior vertices
            Lschur = the.Kron_reduction(filtered_edge_index, filtered_restriction_maps, b_idxs, i_idxs.flatten())
            # reshape the entities that should be scored so they are sized (num_targets*dv, num_entities_to_score)
            xT = torch.transpose(to_score.squeeze(0).squeeze(-1), 0,1)
            # score all the entities over our schur complement Laplacian
            # assign the source vertices to index 0 and the target vertices to index 1 w.r.t. the ordering of the cochain
            costs = the.compute_costs(Lschur, torch.zeros(1), torch.ones(1), sv.flatten(), xT, sv.shape[-2] )
            scores[i] = -costs
        return scores

    def _masked_harmonic_extension(self, h,r,t):
        # TODO
        return self._forward_harmonic_extension(h,r,t)

    def forward(self, h,r,t):
        if self.training:
            if self.training_mask_pct == 0:
                return super().forward(h,r,t)
            return self._masked_harmonic_extension(h,r,t)
        return super().forward(h,r,t)

    def score_masked_head(self, h,r,t,to_score):
        return self._forward_harmonic_extension(t,r,h,to_score)

    def score_masked_tail(self, h,r,t,to_score):
        return self._forward_harmonic_extension(h,r,t,to_score)

    def score_h(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        to_score: HeadRepresentation,
        slice_size: Optional[int] = None,
        slice_dim: int = 1,
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores with optional slicing.

        .. note ::
            At most one of the slice sizes may be not None.

        # TODO: we could change that to slicing along multiple dimensions, if necessary

        :param h: shape: (`*batch_dims`, `*dims`)
            The head representations from the sampler.
        :param r: shape: (`*batch_dims`, `*dims`)
            The relation representations.
        :param t: shape: (`*batch_dims`, `*dims`)
            The tail representations.
        :param h: shape: (`*batch_dims`, `*dims`)
            The head representations to score.
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice. From {0, ..., len(batch_dims)}

        :return: shape: batch_dims
            The scores.
        """
        if slice_size is None:
            return self.score_masked_head(h=h, r=r, t=t, to_score=to_score)

        return torch.cat(
            [
                self.score_masked_head(h=h_batch, r=r_batch, t=t_batch, to_score=to_score)
                for h_batch, r_batch, t_batch in parallel_slice_batches(h, r, t, split_size=slice_size, dim=slice_dim)
            ],
            dim=slice_dim,
        )

def score_t(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        to_score: HeadRepresentation,
        slice_size: Optional[int] = None,
        slice_dim: int = 1,
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores with optional slicing.

        .. note ::
            At most one of the slice sizes may be not None.

        # TODO: we could change that to slicing along multiple dimensions, if necessary

        :param h: shape: (`*batch_dims`, `*dims`)
            The head representations from the sampler.
        :param r: shape: (`*batch_dims`, `*dims`)
            The relation representations.
        :param t: shape: (`*batch_dims`, `*dims`)
            The tail representations.
        :param h: shape: (`*batch_dims`, `*dims`)
            The head representations to score.
        :param slice_size:
            The slice size.
        :param slice_dim:
            The dimension along which to slice. From {0, ..., len(batch_dims)}

        :return: shape: batch_dims
            The scores.
        """
        if slice_size is None:
            return self.score_masked_tail(h=h, r=r, t=t, to_score=to_score)

        return torch.cat(
            [
                self.score_masked_tail(h=h_batch, r=r_batch, t=t_batch, to_score=to_score)
                for h_batch, r_batch, t_batch in parallel_slice_batches(h, r, t, split_size=slice_size, dim=slice_dim)
            ],
            dim=slice_dim,
        )
        

