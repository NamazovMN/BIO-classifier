import os
import torch
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from string import punctuation
from utilities import *
from reader import Reader
from process import ProcessData
from dataset import TokensDataset
from torch.utils.data import DataLoader
from models import parallelModel
from torch.nn import NLLLoss
from torch.optim import Adam, SGD
from train import TrainHuge

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# main_dir = '/content/drive/MyDrive/BIO'
data_dir = '../dataset'
exp_num = 1
epochs = 60
learning_rate = 0.00001

emb_dim = 200
hid_dim = 50
num_inputs = 2
num_layers = 1

train_path = os.path.join(data_dir, 'train-bio.csv')
test_path = os.path.join(data_dir, 'test-bio.csv')
punctuations = punctuation
specials = stopwords.words('english') + ['__END_ESSAY__', '__END_PARAGRAPH__']
sentence_delimiter = ['!', '.', '?', '__END_ESSAY__', '__END_PARAGRAPH__']
reader_train = Reader(train_path, punctuations, specials)
reader_test = Reader(test_path, punctuations, specials)

train_raw_sentences = reader_train.collect_sentences(sentence_delimiter)
print(train_raw_sentences)
test_raw_sentences = reader_test.collect_sentences(sentence_delimiter)
tokens, labels = reader_train.clean_toks()
counter_type = 'paragraph'
collection_all = reader_train.counter_lengths(train_raw_sentences, counter_type)
train_sentences, train_labels, valid_sentences, valid_labels, train_counters, valid_counters = reader_train.split_data(train_raw_sentences, collection_all, sent_type=False)
test_counters = reader_test.counter_lengths(test_raw_sentences, counter_type)
test_sentences, test_labels = reader_test.clean_data(test_raw_sentences)

sentences_data = {
    'train_data': train_sentences,
    'train_labels': train_labels,
    'valid_data': valid_sentences,
    'valid_labels': valid_labels,
    'test_data': test_sentences,
    'test_labels': test_labels
    }
pretty_printer(train_sentences, 'Train Sentences')
pretty_printer(train_labels, 'Train Labels')
pretty_printer(valid_sentences, 'Valid Sentences')
pretty_printer(valid_labels, 'Valid Labels')
pretty_printer(test_sentences, 'Test Sentences')
pretty_printer(test_labels, 'Test Labels')

processor = ProcessData(labels, sentences_data, 1, 1, False, train_counters, valid_counters, test_counters)
ngrams = False
vocabulary = processor.vocabulary
vocabulary_tag = processor.vocabulary_tag
vocabulary_lab = processor.vocabulary_lab
train_enc, train_enc_tag, train_enc_lab = processor.encode_sentences('train') if not ngrams else processor.generate_n_grams('train')
valid_enc, valid_enc_tag, valid_enc_lab = processor.encode_sentences('valid') if not ngrams else processor.generate_n_grams('train')
test_enc, test_enc_tag, test_enc_lab = processor.encode_sentences('test') if not ngrams else processor.generate_n_grams('train')
max_length = len(train_enc[0])
batch_size = max_length

vocabulary_printer(vocabulary, 'Vocabulary of Tokens', 5)
vocabulary_printer(vocabulary_tag, 'Vocabulary of Tags', 5)
vocabulary_printer(vocabulary_lab, 'Vocabulary of Labels', 5)
show_correspondences(train_sentences, train_enc, train_enc_tag, train_enc_lab, 'Train')
show_correspondences(valid_sentences, valid_enc, valid_enc_tag, valid_enc_lab, 'Valid')
show_correspondences(test_sentences, test_enc, test_enc_tag, test_enc_lab, 'Test')
print(f'Maximum length of sentences over all dataset: {max_length}')

train_dataset = TokensDataset(train_enc, train_enc_lab)
train_dataset_tag = TokensDataset(train_enc_tag, train_enc_lab)
valid_dataset = TokensDataset(valid_enc, valid_enc_lab)
valid_dataset_tag = TokensDataset(valid_enc_tag, valid_enc_lab)
test_dataset = TokensDataset(test_enc, test_enc_lab)
test_dataset_tag = TokensDataset(test_enc_tag, test_enc_lab)



vocabulary_size = len(vocabulary)
vocabulary_tag_size = len(vocabulary_tag)
output_size = len(vocabulary_lab)
huge_model = parallelModel(num_inputs, vocabulary_size, vocabulary_tag_size,
                           emb_dim, hid_dim, max_length, num_layers,
                           output_size, vocabulary['<pad>'],
                           vocabulary_tag['<pad>']).to(device)
train_loader = DataLoader(train_dataset, batch_size)
train_tag_loader = DataLoader(train_dataset_tag, batch_size)
valid_loader = DataLoader(valid_dataset, batch_size)
valid_tag_loader = DataLoader(valid_dataset_tag, batch_size)
test_loader = DataLoader(test_dataset, batch_size)
test_tag_loader = DataLoader(test_dataset_tag, batch_size)


# loss_function = CrossEntropyLoss(ignore_index=vocabulary_lab['<pad>'])
loss_function = NLLLoss()
optim = Adam(huge_model.parameters(), lr=learning_rate)
res = 'results'
training_phase = True
if training_phase:
    huge_trainer = TrainHuge(huge_model, epochs, loss_function, optim, exp_num, res, device)
    huge_trainer.train_model(train_loader, valid_loader, train_tag_loader, valid_tag_loader)

else:
    epochs_path = os.path.join('results_and_params', 'model_params_1')
    huge_model_eval = parallelModel(num_inputs, vocabulary_size, vocabulary_tag_size,
                               emb_dim, hid_dim, max_length, num_layers,
                               output_size, vocabulary['<pad>'],
                               vocabulary_tag['<pad>']).to(device)

    model_path = os.path.join(res, epochs_path)
    specific_path = os.path.join(model_path, 'model_1_10')
    huge_eval = torch.load(specific_path)
    huge_eval.eval()
    predictions, targets = compute_precision(huge_eval, test_loader, test_tag_loader, vocabulary_lab['<pad>'], device)

    from sklearn.metrics import f1_score
    f1_macro = f1_score(targets, predictions, average='macro')
    f1_micro = f1_score(targets, predictions, average='micro')
    print(f'f1 macro: {f1_macro}')
    print(f'f1 micro: {f1_micro}')

# print(specific_path)
#
# huge_model_eval = torch.load(specific_path)
# print(huge_model)
# huge_model_eval = huge_model_eval.to(device)