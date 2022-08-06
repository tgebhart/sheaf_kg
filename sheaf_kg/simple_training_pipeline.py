import argparse

import matplotlib.pyplot as plt
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from pykeen.datasets import Nations, FB15k237, WN18RR
from sheaf_kg.models.extension_structured_embedding import ExtensionStructuredEmbedding
from sheaf_kg.regularizers.multisection_regularizers import OrthogonalSectionsRegularizer
from sheaf_kg.models.multisection_structured_embedding import MultisectionStructuredEmbedding
from sheaf_kg.models.multisection_trans_e import MultisectionTransE
from sheaf_kg.triples.triples_factory import ComplexTriplesFactory

DATASET = 'Nations'
# MODEL = 'MultisectionStructuredEmbedding'
MODEL = 'ExtensionStructuredEmbedding'
SAMPLER = 'schlichtkrull'
NUM_EPOCHS = 5
C0_DIM = 50
C1_DIM = 20
NUM_SECTIONS = 1
RANDOM_SEED = 134
REGULARIZER_WEIGHT = 1

model_map = {
    'MultisectionStructuredEmbedding':MultisectionStructuredEmbedding,
    'MultisectionTransE':MultisectionTransE,
    'ExtensionStructuredEmbedding':ExtensionStructuredEmbedding
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

def get_factories(dataset):

    triples_factory = ComplexTriplesFactory

    dstf = find_dataset(dataset)
    train = triples_factory(
        mapped_triples=dstf.training.mapped_triples,
        entity_to_id=dstf.training.entity_to_id,
        relation_to_id=dstf.training.relation_to_id,
        create_inverse_triples=dstf.training.create_inverse_triples,
        metadata=dstf.training.metadata
    )
    validate = triples_factory(
        mapped_triples=dstf.validation.mapped_triples,
        entity_to_id=dstf.validation.entity_to_id,
        relation_to_id=dstf.validation.relation_to_id,
        create_inverse_triples=dstf.validation.create_inverse_triples,
        metadata=dstf.validation.metadata
    )
    test = triples_factory(
        mapped_triples=dstf.testing.mapped_triples,
        entity_to_id=dstf.testing.entity_to_id,
        relation_to_id=dstf.testing.relation_to_id,
        create_inverse_triples=dstf.testing.create_inverse_triples,
        metadata=dstf.testing.metadata
    )
    return train, validate, test

def run(model, dataset, num_epochs, random_seed,
        embedding_dim, c1_dimension=None, num_sections=None,
        sampler=None):

    train_tf, validate_tf, test_tf = get_factories(dataset)
    model_class = find_model(model)
    reg_weight = REGULARIZER_WEIGHT
        
    evaluator = RankBasedEvaluator(
        filtered=True,  # Note: this is True by default; we're just being explicit
    )
    
    model_kwargs = {}
    model_kwargs['C0_dimension'] = embedding_dim
    if num_sections is not None:
        model_kwargs['num_sections'] = num_sections
    if c1_dimension is not None:
        model_kwargs['C1_dimension'] = c1_dimension 
    
    # model_kwargs['training_mask_pct'] = 0.1

    result = pipeline(
        model=model_class,
        model_kwargs=model_kwargs,
        regularizer=OrthogonalSectionsRegularizer,
        regularizer_kwargs={'weight':reg_weight},
        training=train_tf,
        testing=test_tf,
        validation=validate_tf,
        training_kwargs={'sampler':sampler},
        epochs=num_epochs,
        device='cpu',
        random_seed=random_seed,
        evaluator=evaluator,
    )

    # Evaluate your model with not only testing triples,
    # but also filter on validation triples
    ev_results = evaluator.evaluate(
        model=result.model,
        mapped_triples=test_tf.mapped_triples,
        additional_filter_triples=[
            train_tf.mapped_triples,
            validate_tf.mapped_triples,
        ]
    )

    result.plot_losses()
    plt.savefig(f'data/{model}_losses_{reg_weight}.png')
    ev_df = ev_results.to_df()
    print(ev_df)
    ev_df.to_csv(f'data/{model}_losses_{reg_weight}.csv')


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
    training_args.add_argument('--c1-dimension', type=int, default=C1_DIM,
                        help='entity embedding dimension')
    training_args.add_argument('--num-sections', type=int, default=NUM_SECTIONS,
                        help='number of simultaneous sections to learn')
    training_args.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='random seed')
    training_args.add_argument('--model', type=str, required=False, default=MODEL,
                        choices=['MultisectionStructuredEmbedding', 'MultisectionTransE'],
                        help='name of model to train')
    training_args.add_argument('--sampler', type=str, required=False, default=SAMPLER,
                        choices=['complex', 'schlichtkrull' ],
                        help='name of triples factory to use')

    args = parser.parse_args()

    run(args.model, args.dataset, args.num_epochs, args.random_seed,
        args.embedding_dim, c1_dimension=args.c1_dimension, num_sections=args.num_sections,
        sampler=args.sampler)


