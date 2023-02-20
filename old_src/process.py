from collections import Counter
from tqdm import tqdm
from nltk import pos_tag

class ProcessData:
    def __init__(self, labels, sentences_data, window_size=None, window_shift=None, sent_type=True, counter_train=None,
                 counter_valid=None, counter_test=None):

        self.counters = {'train': counter_train, 'valid': counter_valid, 'test': counter_test}
        self.datasets = sentences_data
        self.datasets = self.extract_data(sentences_data)
        self.datasets_essay = self.extract_essays()
        self.window_size = window_size
        self.window_shift = window_shift
        self.vocabulary, self.vocabulary_tag, self.vocabulary_lab = self.build_vocabulary(sentences_data['train_data'],
                                                                                          labels)
        print(self.vocabulary)
        self.max_length = self.get_max_length() if sent_type else self.get_max_essay_length()

    def extract_essays(self):
        data_types = ['train', 'valid', 'test']
        dictionary = dict()
        for each in data_types:
            dictionary[f'{each}_data'], dictionary[f'{each}_labels'], dictionary[
                f'{each}_tags'] = self.genereate_collection_encodings(each)

        return dictionary

    def extract_data(self, dictionary):
        train_tags, test_tags, valid_tags, self.tags = self.generate_tags()
        dictionary['train_tags'] = train_tags
        dictionary['valid_tags'] = valid_tags
        dictionary['test_tags'] = test_tags
        return dictionary

    def get_max_essay_length(self):

        train = max([len(each) for each in self.datasets_essay['train_labels']])
        valid = max([len(each) for each in self.datasets_essay['valid_labels']])
        test = max([len(each) for each in self.datasets_essay['test_labels']])

        return max(train, valid, test)

    def make_tag_sentences(self, set_type, sent=True):
        'idx -> train 0, test 1, valid 2'
        dataset = self.datasets[f'{set_type}_data']

        if sent:
            tag_sentences = list()
            tags = list()

            tqdm_iter = tqdm(dataset, desc=f'{set_type} dataset is tagged', total=len(dataset), leave=True)
            for each_sent in tqdm_iter:
                sentence = [token if token != '' else ' ' for token in each_sent]
                taged_sentence = pos_tag(sentence)
                tag_sentence = [each_token[1] for each_token in taged_sentence]
                assert len(tag_sentence) == len(each_sent)
                tag_sentences.append(tag_sentence)
                tags.extend(tag_sentence)
        return tag_sentences, tags

    def generate_tags(self):
        train_tag_sentences, train_tags = self.make_tag_sentences('train')
        test_tag_sentences, test_tags = self.make_tag_sentences('test')
        valid_tag_sentences, valid_tags = self.make_tag_sentences('valid')
        tags = train_tags + test_tags + valid_tags
        return train_tag_sentences, test_tag_sentences, valid_tag_sentences, tags

    def get_max_length(self):
        train_length = [len(each) for each in self.train]
        test_length = [len(each) for each in self.test]
        valid_length = [len(each) for each in self.valid]

        return max(max(train_length), max(test_length), max(valid_length))

    def build_vocabulary(self, train_sent, labels):
        pure_tokens = list()
        for each_sent in train_sent:
            pure_tokens.extend(each_sent)

        count_tags = Counter(self.tags)
        count_tokens = Counter(pure_tokens)
        count_labels = Counter(labels)
        vocabulary = {token: idx for idx, token in enumerate(count_tokens.keys())}
        vocabulary['<pad>'] = len(vocabulary)
        vocabulary['<unk>'] = len(vocabulary)
        vocabulary_tags = {token: idx for idx, token in enumerate(count_tags.keys())}
        vocabulary_tags['<pad>'] = len(vocabulary_tags)
        vocabulary_lab = {lab: idx for idx, lab in enumerate(count_labels.keys())}
        vocabulary_lab['<pad>'] = len(vocabulary_lab)

        return vocabulary, vocabulary_tags, vocabulary_lab

    def encode_sentences(self, data_type, collection=False):

        dataset = self.datasets if collection else self.datasets_essay
        chosen_set = dataset[f'{data_type}_data']
        chosen_lab_set = dataset[f'{data_type}_labels']
        chosen_tag_set = dataset[f'{data_type}_tags']
        embedded_sentences = list()
        tag_embedded_sentences = list()
        lab_embedded_sentences = list()
        tqdm_iter = tqdm(zip(chosen_set, chosen_tag_set, chosen_lab_set), desc=f'{data_type} data is encoded',
                         total=len(chosen_set), leave=True)
        for each_sentence, each_tag_sentence, each_label in tqdm_iter:
            sentence = each_sentence + (self.max_length - len(each_sentence)) * [None]
            tag_sentence = each_tag_sentence + (self.max_length - len(each_sentence)) * [None]

            labels = each_label + (self.max_length - len(each_label)) * ['<pad>']
            if data_type == 'train':
                emb_sent = [self.vocabulary[token if token != None else '<pad>'] for token in sentence]
            else:
                emb_sent = list()
                for each_token, each_label in zip(sentence, labels):
                    if each_token not in self.vocabulary.keys():
                        emb_sent.append(self.vocabulary['<unk>' if each_token != None else '<pad>'])
                    else:
                        emb_sent.append(self.vocabulary[each_token if each_token != None else '<pad>'])
            tag_emb_sent = [self.vocabulary_tag[token if token != None else '<pad>'] for token in tag_sentence]
            lab_idx = [self.vocabulary_lab[lab] for lab in labels]

            tag_embedded_sentences.append(tag_emb_sent)
            lab_embedded_sentences.append(lab_idx)
            embedded_sentences.append(emb_sent)
        return embedded_sentences, tag_embedded_sentences, lab_embedded_sentences

    def generate_n_grams(self, data_type, collection=False):
        embedded_sentences, tag_embedded_sentences, lab_embedded_sentences = self.encode_sentences(data_type,
                                                                                                   collection)
        w_embedded_sentences = list()
        w_tag_embedded_sentences = list()
        w_lab_embedded_sentences = list()
        for each_sent, each_tag, each_lab in zip(embedded_sentences, tag_embedded_sentences, lab_embedded_sentences):
            cur_sent = list()
            cur_tag_sent = list()
            cur_lab_sent = list()
            for idx in range(0, self.max_length, self.window_shift):
                if idx == self.max_length - self.window_size:
                    break

                cur_sent.extend(each_sent[idx: idx + self.window_size])
                cur_tag_sent.extend(each_lab[idx: idx + self.window_size])
                cur_lab_sent.extend(each_tag[idx: idx + self.window_size])

            w_embedded_sentences.append(cur_sent)
            w_tag_embedded_sentences.append(cur_tag_sent)
            w_lab_embedded_sentences.append(cur_lab_sent)
        return w_embedded_sentences, w_tag_embedded_sentences, w_lab_embedded_sentences

    def genereate_collection_encodings(self, data_type):
        data = self.datasets[f'{data_type}_data']
        labels = self.datasets[f'{data_type}_labels']
        tags = self.datasets[f'{data_type}_tags']

        init_idx = 0

        data_essays = list()
        labels_essays = list()
        tags_essays = list()

        for each in self.counters[data_type]:
            end_idx = init_idx + each
            current_essay = list()
            current_labels = list()
            current_tags = list()
            for k in range(init_idx, end_idx):
                current_essay.extend(data[k])
                current_labels.extend(labels[k])
                current_tags.extend(tags[k])

            init_idx = end_idx
            data_essays.append(current_essay)
            labels_essays.append(current_labels)
            tags_essays.append(current_tags)
        return data_essays, labels_essays, tags_essays





