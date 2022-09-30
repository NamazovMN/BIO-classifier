class Reader:
    def __init__(self, path, punct, specials):
        self.path = path
        self.specials = specials
        self.punct = punct

    def read_lines(self) -> list:
        '''
        The function is used in order to collect raw data in clean way
        :return: raw_data is list of tuples, where each tuple includes token and its label
        '''
        file = open(self.path, 'r')
        raw_data = list()
        for each_line in file:
            token, label = each_line.split('\t')
            label = label.split('\n')[0]
            if token != '' and token != ' ' and token != '\n':
                raw_data.append((token, label))
            else:
                pass

        return raw_data

    def collect_sentences(self, delimiter) -> list:
        '''
        The function is for collecting sentences from the raw tokens data
        :param delimiter: list of punctuations define where sentence ends
        :return: list of sentences where each sentence is list of tuples
        '''
        raw_data = self.read_lines()
        sentence = list()
        sentences = list()
        for idx, each_data in enumerate(raw_data):
            sentence.append(each_data)
            if idx + 1 == len(raw_data):
                sentences.append(sentence)
            elif each_data[0] in delimiter:
                if raw_data[idx + 1][0] == '__END_PARAGRAPH__':
                    pass
                elif raw_data[idx + 1][0] == '__END_ESSAY__':
                    pass
                else:
                    sentences.append(sentence)
                    sentence = list()

        return sentences

    def counter_lengths(self, sentences, collection_type='essay') -> list:
        '''
        The function is used in order to collect information about number of sentences per essay or paragraph
        :param sentences: list of sentences, where each sentence is list of tuples
        :param collection_type: string object defines whether we need information per essay or paragraph
        :return: list of (number of sentences) per essay or paragraph
        '''
        delimiter = ('__END_ESSAY__', 'O') if collection_type == 'essay' else ('__END_PARAGRAPH__', 'O')
        collect_count = list()
        sent_count = 0
        for each_sentence in sentences:
            sent_count += 1
            if delimiter in each_sentence:
                collect_count.append(sent_count)
                sent_count = 0
        return collect_count

    def clean_data(self, collection) -> tuple:
        '''
        The function cleans each sentence from punctuations and stop words in english language
        :param collection: list of sentences
        :return: returns sentences (where each element is token) and list of label sentences (where each element is label)
        '''
        token_collection = list()
        label_collection = list()
        for each_sentence in collection:
            token_sentence = list()
            label_sentence = list()
            for each_token, each_label in each_sentence:
                if each_token not in self.punct and each_token not in self.specials:
                    token_sentence.append(each_token.lower())
                    label_sentence.append(each_label)
            token_sentence.append('<end_s>')
            label_sentence.append('O')
            token_collection.append(token_sentence)
            label_collection.append(label_sentence)
        return token_collection, label_collection

    def clean_toks(self):
        tokens = self.read_lines()
        clean_tokens = list()
        clean_labels = list()
        for each_token, each_label in tokens:
            if each_token not in self.punct and each_token not in self.specials:
                clean_tokens.append(each_token.lower())
                clean_labels.append(each_label)
        return clean_tokens, clean_labels

    def split_data(self, sentences, counters=None, percentage=0.8, sent_type=True):
        '''
        The function is used to split dataset into train and validation data
        :param sentences: list of sentences per essay or paragraph according to the type was defined
        :param counters: number of essays or paragraphs in the dataset. Will be used if the task will not be sentence
                         based classification
        :param percentage: percentage of split -> Train/Validation ratio will be percentage/1-percentage
        :param sent_type: defines whether classification task will be done sentence based or not
        :return: train_sentences: list of train sentences
                 train_labels: list of train labels (sentence of labels)
                 valid_sentences: list of validation sentences
                 valid_labels: list of validation labels (sentence of labels)
                 train_counters: number of paragraphs or essays in train dataset
                 valid_counters: number of paragraphs or essays in validation dataset
        '''
        train_length = 0
        sentences, labels = self.clean_data(sentences)
        length = int(len(sentences) * percentage) if sent_type else int(len(counters) * percentage)
        if not sent_type:
            num_of_sentences = 0
            for idx, each_length in enumerate(counters):
                num_of_sentences += each_length
                if idx == length - 1:
                    train_length = num_of_sentences
                    break
        train_counters = counters[:length] if counters != None else list()
        valid_counters = counters[length::] if counters != None else list()
        train_sentences = sentences[:train_length]
        valid_sentences = sentences[train_length::]
        train_labels = labels[:train_length]
        valid_labels = labels[train_length::]
        return train_sentences, train_labels, valid_sentences, valid_labels, train_counters, valid_counters