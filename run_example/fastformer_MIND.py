# @Time   : 2022/7/15
# @Author : Victor Chen
# @Email  : vic4code@gmail.com


"""
news recommendation example
========================
Here is the sample code for running news recommendation benchmarks using RecBole based on Fastformer.

For the data preparation, you need to follow the description first - https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/MIND.md
"""

import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color

import os, sys
sys.path.append(os.path.abspath(".."))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='Fastformer', help='Model for news rec.')
    parser.add_argument('--dataset', '-d', type=str, default='mind', help='Benchmarks for news rec.')
    parser.add_argument('--validation', action='store_true', help='Whether evaluating on validation set (split from train set), otherwise on test set.')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = get_args()

    # configurations initialization
    config_dict = {
        'neg_sampling': None,
        'benchmark_filename': ['small_train', 'small_dev'],
        'metrics': ['Recall', 'MRR'],
    }

    config = Config(model=args.model, dataset=f'{args.dataset}', config_dict=config_dict, config_file_list=['../recbole/properties/dataset/mind.yaml'])
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    # train_dataset, test_dataset = dataset.build()
    # if args.validation:
    #     train_dataset.shuffle()
    #     new_train_dataset, new_test_dataset = train_dataset.split_by_ratio([1 - args.valid_portion, args.valid_portion])
    #     train_data = get_dataloader(config, 'train')(config, new_train_dataset, None, shuffle=True)
    #     test_data = get_dataloader(config, 'test')(config, new_test_dataset, None, shuffle=False)
    # else:
    #     train_data = get_dataloader(config, 'train')(config, train_dataset, None, shuffle=True)
    #     test_data = get_dataloader(config, 'test')(config, test_dataset, None, shuffle=False)

    # # model loading and initialization
    # model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    # logger.info(model)

    # # trainer loading and initialization
    # trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # # model training and evaluation
    # test_score, test_result = trainer.fit(
    #     train_data, test_data, saved=True, show_progress=config['show_progress']
    # )

    # logger.info(set_color('test result', 'yellow') + f': {test_result}')
