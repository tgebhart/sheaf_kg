# Knowledge Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph Embedding

Models are written to be compatible with the
[Pykeen package](https://pykeen.readthedocs.io/en/stable/).

Download BetaE data from [here](http://snap.stanford.edu/betae/KG_data.zip) and 
put it in the top-level directory, keeping the folder name `KG_data`. 

## Recreate Figures

To recreate the experiment data, first run each of `naive_experiments.py`, `se_experiments.py`, `transe_experiments.py`, 
`translational_experiments.py`, `se_orthogonal_experiments.py`, and `translational_orthogonal_experiments.py`. 

The results of the above runner scripts  will create data in an opinionated subdirectory structure within the `data` directory.
This data can then be read in and used to recreate the paper figures. 
See `format_results.ipynb` for the figure creation code.  

