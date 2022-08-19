# @Time   : 2022/7/15
# @Author : Victor Chen
# @Email  : vic4code@gmail.com


"""
news recommendation example
========================
Here is the sample code for running news recommendation benchmarks using RecBole based on Fastformer.

For the data preparation, you need to follow the description first - https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/MIND.md
"""

import os
import requests
import logging
import math
import zipfile
import argparse
from logging import getLogger
import numpy as np
import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, dataset
from recbole.data.dataset import Dataset
from recbole.data.utils import get_dataloader
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color, FeatureSource, FeatureType
from recbole.trainer.trainer import Trainer
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer


log = logging.getLogger(__name__)


# Prepare pretrained Glove and convert news words to word embeddings matrix
class NewsRecDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset']
        self.logger = getLogger()
        self._from_scratch()

    def _from_scratch(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """
        self.logger.debug(set_color(f'Loading {self.__class__} from scratch.', 'green'))

        self._get_preset()
        self._get_field_from_config()
        self._load_data(self.dataset_name, self.dataset_path)
        self._init_alias()

        # Convert news words to word embeddings matrix
        self.normed_news_words = self.get_words()
        self.news_feature_path, self.word_embeddings_path = self.generate_embeddings(self.normed_news_words)
        self.news_feature = self.load_glove_matrix(self.news_feature_path)

        self._data_processing()


    # def __init__(self, config):
    #     super().__init__(config)

        
        # news_feature_path, word_embeddings_path = self.generate_embeddings(normed_news_words)
        
        # # Remap words with glove
        # self._remap_news_words(news_feature_path, word_embeddings_path)
        

    def _remap(self, remap_list):
        """Remap tokens using :meth:`pandas.factorize`.

        Args:
            remap_list (list): See :meth:`_get_remap_list` for detail.
        """
        if len(remap_list) == 0:
            return
        tokens, split_point = self._concat_remaped_tokens(remap_list)
        new_ids_list, mp = pd.factorize(tokens)
        new_ids_list = np.split(new_ids_list + 1, split_point)
        mp = np.array(['[PAD]'] + list(mp))
        token_id = {t: i for i, t in enumerate(mp)}

        for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
            if field not in self.field2id_token:
                self.field2id_token[field] = mp
                self.field2token_id[field] = token_id
            if ftype == FeatureType.TOKEN:
                feat[field] = new_ids
            elif ftype == FeatureType.TOKEN_SEQ:
                self.field2id_token[field] = [token for token in self.normed_news_words.keys()]

                glove_token_id = []
                for line in feat[field]:
                    ids = []
                    for token in line:
                        ids += self.news_feature(token)
                    glove_token_id.append(ids)

                feat[field] = glove_token_id
                print(self.field2id_token[field][:5])
                # feat[field] = _token2_ids()


            # print(mp[:5])
            # print(list(token_id.items())[:5])
    
    # def _token2_ids(self):

    def _remap_news_words(self, news_feature_path, word_embeddings_path):
        """Remap news tokens using glove`.
        """
        embedding_matrix, exist_word = self.load_glove_matrix(news_feature_path, word_embeddings_path)

        print(embedding_matrix.shape)
        print(len(exist_word))
        # if len(remap_list) == 0:
        #     return
        # tokens, split_point = self._concat_remaped_tokens(remap_list)
        # new_ids_list, mp = pd.factorize(tokens)
        # new_ids_list = np.split(new_ids_list + 1, split_point)
        # mp = np.array(['[PAD]'] + list(mp))
        # token_id = {t: i for i, t in enumerate(mp)}

        # for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
        #     if field not in self.field2id_token:
        #         self.field2id_token[field] = mp
        #         self.field2token_id[field] = token_id
        #     if ftype == FeatureType.TOKEN:
        #         feat[field] = new_ids

        
    def _load_data(self, token, dataset_path):
        """Load features.
        Firstly load interaction features, then user/item features optionally,
        finally load additional features if ``config['additional_feat_suffix']`` is set.
        Args:
            token (str): dataset name.
            dataset_path (str): path of dataset dir.
        """
        if not os.path.exists(dataset_path):
            self._download()
        self._load_inter_feat(token, dataset_path)
        self.user_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.USER, 'uid_field')
        self.item_feat = self._load_news(token, dataset_path)
        self._load_additional_feat(token, dataset_path)


    def _load_news(self, token, dataset_path):
        if self.benchmark_filename_list is None:
            item_feat_path = os.path.join(dataset_path, f'{token}.item')
            if not os.path.isfile(item_feat_path):
                item_feat = None
                self.logger.debug(f'[{feat_path}] not found, [{source.value}] features are not loaded.')
            else:
                item_feat = self._load_feat(item_feat_path, FeatureSource.ITEM)
                self.logger.debug(f'Item feature loaded successfully from [{item_feat_path}].')

        else:
            sub_item_feats = []
            for filename in self.benchmark_filename_list:
                file_path = os.path.join(dataset_path, f'{token}.{filename}.item')
                if os.path.isfile(file_path):
                    temp = self._load_feat(file_path, FeatureSource.ITEM)
                    sub_item_feats.append(temp)
                else:
                    raise ValueError(f'File {file_path} not exist.')
            item_feat = pd.concat(sub_item_feats, ignore_index=True)
        
        return item_feat


    def _read_news(self, news_words, tokenizer):

        lines = self.item_feat['title']
        for line in lines:
            for word in line:
                splitted = word.strip("\n").split("\t")
                if splitted and splitted[0] not in news_words:
                    news_words[splitted[0]] = tokenizer.tokenize(splitted[0].lower())

        return news_words


    def unzip_file(self, zip_src, dst_dir, clean_zip_file=False):
        """Unzip a file
        Args:
            zip_src (str): Zip file.
            dst_dir (str): Destination folder.
            clean_zip_file (bool): Whether or not to clean the zip file.
        """
        fz = zipfile.ZipFile(zip_src, "r")
        for file in fz.namelist():
            fz.extract(file, dst_dir)
        if clean_zip_file:
            os.remove(zip_src)


    def get_words(self):
        """Load words
        Returns:
            dict: Words dictionaries.
        """
        news_words = {}
        tokenizer = RegexpTokenizer(r"\w+")
        normed_news_words = self._read_news(
            news_words, tokenizer
        )

        return normed_news_words


    def maybe_download(self, url, filename=None, work_directory=".", expected_bytes=None):
        """Download a file if it is not already downloaded.
        Args:
            filename (str): File name.
            work_directory (str): Working directory.
            url (str): URL of the file to download.
            expected_bytes (int): Expected file size in bytes.
        Returns:
            str: File path of the file downloaded.
        """
        if filename is None:
            filename = url.split("/")[-1]
        os.makedirs(work_directory, exist_ok=True)
        filepath = os.path.join(work_directory, filename)
        if not os.path.exists(filepath):
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                log.info(f"Downloading {url}")
                total_size = int(r.headers.get("content-length", 0))
                block_size = 1024
                num_iterables = math.ceil(total_size / block_size)
                with open(filepath, "wb") as file:
                    for data in tqdm(
                        r.iter_content(block_size),
                        total=num_iterables,
                        unit="KB",
                        unit_scale=True,
                    ):
                        file.write(data)
            else:
                log.error(f"Problem downloading {url}")
                r.raise_for_status()
        else:
            log.info(f"File {filepath} already downloaded")
        if expected_bytes is not None:
            statinfo = os.stat(filepath)
            if statinfo.st_size != expected_bytes:
                os.remove(filepath)
                raise IOError(f"Failed to verify {filepath}")

        return filepath


    def download_and_extract_glove(self, dest_path):
        """Download and extract the Glove embedding
        Args:
            dest_path (str): Destination directory path for the downloaded file
        Returns:
            str: File path where Glove was extracted.
        """
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        filepath = self.maybe_download(url=url, work_directory=dest_path)
        glove_path = os.path.join(dest_path, "glove")
        self.unzip_file(filepath, glove_path, clean_zip_file=False)
        return glove_path


    def generate_embeddings(
        self,
        news_words,
        max_sentence=10,
        word_embedding_dim=100,
    ):
        """Generate embeddings.
        Args:
            news_words (dict): News word dictionary.
            max_sentence (int): Max sentence size.
            word_embedding_dim (int): Word embedding dimension.
        Returns:
            str, str: File paths to news, and word embeddings.
        """
        embedding_dimensions = [50, 100, 200, 300]
        if word_embedding_dim not in embedding_dimensions:
            raise ValueError(
                f"Wrong embedding dimension, available options are {embedding_dimensions}"
            )

        logger.info("Downloading glove...")
        data_path = os.path.join(self.config['data_path'], 'word_embeddings')
        if not os.path.exists:
            os.makedirs(data_path)
        glove_path = self.download_and_extract_glove(data_path)

        word_set = set()
        word_embedding_dict = {}
        entity_embedding_dict = {}

        logger.info(f"Loading glove with embedding dimension {word_embedding_dim}...")
        glove_file = "glove.6B." + str(word_embedding_dim) + "d.txt"
        fp_pretrain_vec = open(os.path.join(glove_path, glove_file), "r", encoding="utf-8")
        for line in fp_pretrain_vec:
            linesplit = line.split(" ")
            word_set.add(linesplit[0])
            word_embedding_dict[linesplit[0]] = np.asarray(list(map(float, linesplit[1:])))
        fp_pretrain_vec.close() 

        logger.info("Generating word indexes...")
        word_dict = {}
        word_index = 1
        news_word_string_dict = {}
        for doc_id in news_words:
            news_word_string_dict[doc_id] = [0 for n in range(max_sentence)]
            
            for i in range(len(news_words[doc_id])):
                if news_words[doc_id][i] in word_embedding_dict:
                    if news_words[doc_id][i] not in word_dict:
                        word_dict[news_words[doc_id][i]] = word_index
                        word_index = word_index + 1
                        news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
                    else:
                        news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
                    
                if i == max_sentence - 1:
                    break

        logger.info("Generating word embeddings...")
        word_embeddings = np.zeros([word_index, word_embedding_dim])
        for word in word_dict:
            word_embeddings[word_dict[word]] = word_embedding_dict[word]

        news_feature_path = os.path.join(data_path, "doc_feature.txt")
        logger.info(f"Saving word features in {news_feature_path}")
        fp_doc_string = open(news_feature_path, "w", encoding="utf-8")
        for doc_id in news_word_string_dict:
            fp_doc_string.write(
                doc_id
                + " "
                + ",".join(list(map(str, news_word_string_dict[doc_id])))
                + "\n"
            )

        word_embeddings_path = os.path.join(
            data_path, "word_embeddings_5w_" + str(word_embedding_dim) + ".npy"
        )
        logger.info(f"Saving word embeddings in {word_embeddings_path}")
        np.save(word_embeddings_path, word_embeddings)

        return news_feature_path, word_embeddings_path


    def load_glove_matrix(self, news_feature_path, word_embeddings_path):
        """Load pretrained embedding metrics of words in word_dict
        Args:
            path_emb (string): Folder path of downloaded glove file
            word_dict (dict): word dictionary
            word_embedding_dim: dimention of word embedding vectors
        Returns:
            numpy.ndarray, list: pretrained word embedding metrics, words can be found in glove files
        """

        embedding_matrix = np.load(word_embeddings_path)
        
        word2ids = []
        with open(os.path.join(path_emb, f"glove.6B.{word_embedding_dim}d.txt"), "rb") as f:
            for l in tqdm(f):  # noqa: E741 ambiguous variable name 'l'
                l = l.split()  # noqa: E741 ambiguous variable name 'l'
                word = l[0].decode()
                if len(word) != 0:
                    if word in word_dict:
                        wordvec = [float(x) for x in l[1:]]
                        index = word_dict[word]
                        embedding_matrix[index] = np.array(wordvec)
                        exist_word.append(word)

        return embedding_matrix, exist_word
            

    def word_tokenize(self, sent):
        """Tokenize a sententence
        Args:
            sent: the sentence need to be tokenized
        Returns:
            list: words in the sentence
        """

        # treat consecutive words or special punctuation as words
        pat = re.compile(r"[\w]+|[.,!?;|]")
        if isinstance(sent, str):
            return pat.findall(sent.lower())
        else:
            return []


class NewsRecTrainer(Trainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.
    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.
    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.
    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.tokenizer = spm.SentencePieceProcessor()

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch
        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.
        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='Fastformer', help='Model for news rec.')
    parser.add_argument('--dataset', '-d', type=str, default='mind', help='Benchmarks for news rec.')
    parser.add_argument('--validation', action='store_true', default=None, help='Whether evaluating on validation set (split from train set), otherwise on test set.')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='ratio of validation set.')
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = get_args()

    # configurations initialization
    config_dict = {
        'benchmark_filename': ['small_train', 'small_dev'],
        'epochs': 300,
        'train_batch_size': 64,
        'learner': 'adam',
        'learning_rate': 0.001,
        'neg_sampling': None,
        'uniform': 1,
        'eval_step': 1,
        'stopping_step': 10,
        # clip_grad_norm:  {'max_norm': 5, 'norm_type': 2}
        'weight_decay': 0.0,
        'loss_decimal_place': 4,
        'require_pow': False,

        # evaluation settings
        # 'eval_args':{ 
        #     'split': {'RS':[0.8,0.1,0.1]},
        #     'group_by': 'user',
        #     'order': 'RO',
        #     'mode': 'full'
        # },
        # 'repeatable': False,
        'metrics': ["Recall","MRR","NDCG","Hit","Precision"],
        # 'topk': [10],
        # 'valid_metric': 'MRR@10',
        # 'valid_metric_bigger': True,
        # 'eval_batch_size': 4096,
        # 'metric_decimal_place': 4
    }

    # config dict is for overall settings; config_file_list is for the other settings like dataset, model settings
    config = Config(model=args.model, dataset=f'{args.dataset}', config_dict=config_dict, config_file_list=['../recbole/properties/dataset/mind.yaml'])
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = NewsRecDataset(config)
    print(dataset.item_feat)
    # print(dataset.field2id_token)
    # print(dataset.field2token_id)
    # dataset.init_news()
    logger.info(dataset)

    # dataset splitting
    # train_dataset, test_dataset = dataset.build()
    # if args.validation:
    #     print('val')
    #     train_dataset.shuffle()
    #     new_train_dataset, new_test_dataset = train_dataset.split_by_ratio([1 - args.valid_portion, args.valid_portion])
    #     train_data = get_dataloader(config, 'train')(config, new_train_dataset, None, shuffle=True)
    #     test_data = get_dataloader(config, 'test')(config, new_test_dataset, None, shuffle=False)
    # else:
    #     train_data = get_dataloader(config, 'train')(config, train_dataset, None, shuffle=True)
    #     test_data = get_dataloader(config, 'test')(config, test_dataset, None, shuffle=False)

    # print(train_data._next_batch_data())
    # # # model loading and initialization
    # model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    # logger.info(model)

    # # trainer loading and initialization
    # trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # # model training and evaluation
    # test_score, test_result = trainer.fit(
    #     train_data, test_data, saved=True, show_progress=config['show_progress']
    # )

    # logger.info(set_color('test result', 'yellow') + f': {test_result}')
