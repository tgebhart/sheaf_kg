import argparse
import itertools
from sheaf_kg.betae_training_pipeline import run

num_epochs = 250
random_seed = 134
datasets = ['NELL', 'FB15k-237']
models = ['BetaeExtensionStructuredEmbedding']
cochain_dims = [[32,32], [32,16], [32,8], [8,8], [16,16], [32,32], [64,64]]
sections = [1, 16, 32]
reg_weights = [0, 1e-2, 1e-1, 1]
parametrizations = [None]

training_batch_size = 2048
evaluation_batch_size = 50
slice_size = 3000

def filter_irrelevant(x):
  if x[3] == 1 and x[4] > 0:
    return False
  else:
    return True

def main(st=0):

  experiments = itertools.product(models, datasets, cochain_dims, sections, reg_weights, parametrizations)
  # filter out experiments from product which do not make sense
  experiments = list(filter(filter_irrelevant, experiments))

  for i in range(st, len(experiments)):
      e = experiments[i]
      print(f'running experiment {i}/{len(experiments)}', e)
      model = e[0]
      dataset = e[1]
      embedding_dim = e[2][0]
      c1_dimension = e[2][1]
      num_sections = e[3]
      reg_weight = e[4]
      parametrization = e[5]
      run(model, dataset, num_epochs, random_seed,
          embedding_dim, c1_dimension=c1_dimension, num_sections=num_sections, reg_weight=reg_weight,
          parameterization=parametrization,
          training_batch_size=training_batch_size, evaluation_batch_size=evaluation_batch_size, slice_size=slice_size)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='simple PyKeen training pipeline')
  # Training Hyperparameters
  training_args = parser.add_argument_group('training')
  training_args.add_argument('--start', type=int, default=0,
                      help='restart at this experiment')
  args = parser.parse_args()
  main(st=args.start)