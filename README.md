# Knowledge Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph Embedding

Models are written to be compatible with the
[Pykeen package](https://pykeen.readthedocs.io/en/stable/).

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
