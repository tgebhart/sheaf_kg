# A Sheaf Theoretic Framework for Knoweldge Graph Embedding

Models are written to be compatible with the
[Pykeen package](https://pykeen.readthedocs.io/en/stable/).

The primary Sheaf-specific models may be found in `sheaf_kg/sheafE_models.py`.
These models can be trained via `sheaf_kg/train_sheafE.py`. This function 
currently uses a relative path to save data in the data folder, so run from 
inside `sheaf_kg/sheaf_kg` for now. The rest of the files are either deprecated 
or in progress of being refactored. The notebooks are mostly scratch work at this 
point. 

Dependencies:
- pykeen
- pytorch
- numpy
- pandas
- scipy
