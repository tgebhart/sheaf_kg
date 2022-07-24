import argparse

import matplotlib.pyplot as plt
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from pykeen.datasets import Nations, FB15k237, WN18RR
from sheaf_kg.regularizers.multisection_regularizers import OrthogonalSectionsRegularizer
from sheaf_kg.models.multisection_structured_embedding import MultisectionStructuredEmbedding
from sheaf_kg.models.multisection_trans_e import MultisectionTransE

DATASET = 'Nations'
MODEL = 'MultisectionStructuredEmbedding'
NUM_EPOCHS = 25
C0_DIM = 50
NUM_SECTIONS = 5
RANDOM_SEED = 134
REGULARIZER_WEIGHT = 1

model_map = {
    'MultisectionStructuredEmbedding':MultisectionStructuredEmbedding,
    'MultisectionTransE':MultisectionTransE
}

dataset_map = {
    'Nations': Nations(),
    'FB15k-237': FB15k237(),
    'WN18RR': WN18RR()
}

def find_model(model):
    if model in model_map:
        return model_map[model]
    raise ValueError(f'model {model} not known')

def find_dataset(dataset):
    if dataset in dataset_map:
        return dataset_map[dataset]
    raise ValueError(f'dataset {dataset} not known')


def run(model, dataset, num_epochs, random_seed,
        embedding_dim, num_sections):

    dataset_instance = find_dataset(dataset)
    model_class = find_model(model)
    reg_weight = REGULARIZER_WEIGHT
        
    evaluator = RankBasedEvaluator(
        filtered=True,  # Note: this is True by default; we're just being explicit
    )

    result = pipeline(
        # model=MultisectionSE,
        # model_kwargs={'C0_dimension':50, 'num_sections':5, 'C1_dimension':25},
        model=model_class,
        model_kwargs={'C0_dimension':embedding_dim, 'num_sections':num_sections},
        regularizer=OrthogonalSectionsRegularizer,
        regularizer_kwargs={'weight':reg_weight},
        dataset=dataset_instance,
        epochs=num_epochs,
        device='gpu',
        random_seed=random_seed,
        evaluator=evaluator,
    )

    # Evaluate your model with not only testing triples,
    # but also filter on validation triples
    ev_results = evaluator.evaluate(
        model=result.model,
        mapped_triples=dataset_instance.testing.mapped_triples,
        additional_filter_triples=[
            dataset_instance.training.mapped_triples,
            dataset_instance.validation.mapped_triples,
        ]
    )

    result.plot_losses()
    plt.savefig(f'data/{model}_losses_{reg_weight}.png')
    print(ev_results.to_df())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simple PyKeen training pipeline')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default=DATASET,
                        choices=['WN18RR','FB15k-237', 'Nations'],
                        help='dataset (default: Nations)')
    training_args.add_argument('--num-epochs', type=int, default=NUM_EPOCHS,
                        help='number of training epochs')
    training_args.add_argument('--embedding-dim', type=int, default=C0_DIM,
                        help='entity embedding dimension')
    training_args.add_argument('--num-sections', type=int, default=NUM_SECTIONS,
                        help='number of simultaneous sections to learn')
    training_args.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='random seed')
    training_args.add_argument('--model', type=str, required=False, default=MODEL,
                        choices=['MultisectionStructuredEmbedding', 'MultisectionTransE'],
                        help='name of model to train')

    args = parser.parse_args()

    run(args.model, args.dataset, args.num_epochs, args.random_seed,
        args.embedding_dim, args.num_sections)


