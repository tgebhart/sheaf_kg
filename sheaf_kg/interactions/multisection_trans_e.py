from pykeen.nn.modules import Interaction

class MultisectionTranslationalInteraction(Interaction):

    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    relation_shape = (
        'ed',  # Corresponds to $\mathbf{M}^{head}_j$
        'ed',  # Corresponds to $\mathbf{M}^{tail}_j$
        'dc',
    )
    # entity to be of shape (#cochain_dimension, #num_cochains)
    entity_shape = ('dc',)

    def forward(self, h, r, t):
        # Since the relation_shape is more than length 1, the r value is given as a sequence
        # of the representations defined there. You can use tuple unpacking to get them out
        r_h, r_t, r_c = r
        h_proj = r_h @ h
        t_proj = r_t @ t
        closs = -(h_proj + r_c - t_proj).norm(p=self.p, dim=-2).sum(dim=-1)
    
        return closs


class MultisectionTransEInteraction(Interaction):

    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    relation_shape = (
        'dc',  # Corresponds to $\mathbf{M}^{head}_j$
    )
    # entity to be of shape (#cochain_dimension, #num_cochains)
    entity_shape = ('dc',)

    def forward(self, h, r, t):
        closs = -(h + r - t).norm(p=self.p, dim=-2).sum(dim=-1) 
        return closs