# Knowledge Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph Embedding

Models are written to be compatible with the
[Pykeen package](https://pykeen.readthedocs.io/en/stable/).

<<<<<<< HEAD
Download BetaE data from (here)[http://snap.stanford.edu/betae/KG_data.zip] and 
put it in the top-level directory, keeping the folder name `KG_data`. 

## Recreate Figures

- Figure 2 (NELL): 
- Figure 4 (NELL):
- Figure 5 (NELL):
- Figure 6 (FB15k-237):
- Figure 7 (FB15k-237):
- Figure 8 (FB15k-237):

- Table 1 (NELL & FB15k-237):
- Table 2 (NELL & FB15k-237):

=======
Dependencies:
- pykeen
- pytorch
- numpy
- pandas
- scipy
- cvxpy
- cvxpylayers

The primary Sheaf-specific models may be found in `sheaf_kg/sheafE_models.py`.
These models can be trained via `sheaf_kg/train_sheafE_sampler_pykeen.py`. This function
currently uses a relative path to load data saved in the data folder, so run from
inside `sheaf_kg/sheaf_kg` for now.

For example, to run the Shv model on FB15k-237, we would run:
```
$ python train_sheafE_sampler_pykeen.py --dataset FB15k-237 --model Multisection \
--embedding-dim <emb_dim> --edge-stalk-dim <esdim> --num-epochs <n_epochs>  \
--test-extension --complex-solver schur --lbda 0 --loss MarginRankingLoss \
--num-sections <n_sections> --orthogonal --alpha-orthogonal <alpha>
```

Or the ShvT model:
```
$ python train_sheafE_sampler_pykeen.py --dataset FB15k-237 --model Translational \
--embedding-dim <emb_dim> --edge-stalk-dim <esdim> --num-epochs <n_epochs>  \
--test-extension --complex-solver cvx --lbda 0 --loss MarginRankingLoss \
--num-sections <n_sections> --orthogonal --alpha-orthogonal <alpha>
```

The data used for the experiments are located in `data/FB15k-237-betae` and
`data/NELL995-betae` which are reformatted versions of the data used in
https://arxiv.org/abs/2010.11465.
>>>>>>> main
