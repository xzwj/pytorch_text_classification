from torchtext import data
from torchtext import datasets
import torch
import os
import dill

SPLIT_RATIO = 0.8
MAX_VOCAB_SIZE = 25_000
PRETRAIN_EMB = "glove.6B.300d"


class IMDB_(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)


class IMDB_Dataloader(object):
    def __init__(self, batch_size, device, data_path='./data'):
        # define field
        self.TEXT = data.Field(tokenize='spacy', # tokenize=nltk.word_tokenize
                                tokenizer_language="en_core_web_sm", 
                                include_lengths=True,
                                batch_first=True,
                                )
        self.LABEL = data.LabelField(dtype=torch.float)

        if self._is_dataset_preprocessed(data_path):
            # load TEXT, LABEL, and examples
            train_data, val_data, test_data = self._load(data_path)
        else:
            # split dataset
            train_data, test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)
            train_data, val_data = train_data.split(split_ratio=SPLIT_RATIO)

            # build vocab
            self.TEXT.build_vocab(train_data,
                                max_size=MAX_VOCAB_SIZE,
                                vectors=PRETRAIN_EMB,
                                unk_init=torch.Tensor.normal_,
                                )
            self.LABEL.build_vocab(train_data)

            # save TEXT, LABEL, and examples
            self._save(data_path, train_data, val_data, test_data)

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(val_data)}')
        print(f'Number of testing examples: {len(test_data)}')
        print(f"Unique tokens in TEXT vocabulary: {len(self.TEXT.vocab)}")
        print(f"Unique tokens in LABEL vocabulary: {len(self.LABEL.vocab)}")

        # create iterator
        self.train_iterator, self.val_iterator, self.test_iterator = \
                                            data.BucketIterator.splits(
                                                (train_data, val_data, test_data),
                                                batch_size=batch_size,
                                                device=device,
                                                )


    def _is_dataset_preprocessed(self, data_path):
        file_name_list = ['text_field.pt', 'label_field.pt', 'train_examples.pt', 
                            'val_examples.pt', 'test_examples.pt']
        for file_name in file_name_list:
            if not os.path.exists(os.path.join(data_path, file_name)):
                return False
        return True


    def _save(self, data_path, train_data, val_data, test_data):
        torch.save(self.TEXT, os.path.join(data_path, 'text_field.pt'), pickle_module=dill)
        torch.save(self.LABEL, os.path.join(data_path, 'label_field.pt'), pickle_module=dill)
        torch.save(train_data.examples, os.path.join(data_path, 'train_examples.pt'), pickle_module=dill)
        torch.save(val_data.examples, os.path.join(data_path, 'val_examples.pt'), pickle_module=dill)
        torch.save(test_data.examples, os.path.join(data_path, 'test_examples.pt'), pickle_module=dill)


    def _load(self, data_path):
        # load fields and examples
        self.TEXT = torch.load(os.path.join(data_path, 'text_field.pt'), pickle_module=dill)
        self.LABEL = torch.load(os.path.join(data_path, 'label_field.pt'), pickle_module=dill)
        train_examples = torch.load(os.path.join(data_path, 'train_examples.pt'), pickle_module=dill)
        val_examples = torch.load(os.path.join(data_path, 'val_examples.pt'), pickle_module=dill)
        test_examples = torch.load(os.path.join(data_path, 'test_examples.pt'), pickle_module=dill)

        # recreate dataset
        fields = [('text', self.TEXT), ('label', self.LABEL)]
        train_data = IMDB_(examples=train_examples, fields=fields)
        val_data = IMDB_(examples=val_examples, fields=fields)
        test_data = IMDB_(examples=test_examples, fields=fields)

        return train_data, val_data, test_data

        


if __name__ == '__main__':
    # Test function `imdb_dataloader`
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_iters = IMDB_Dataloader(10, device, './data')
    # print(dl['train'])
    # print(dl['test'])






