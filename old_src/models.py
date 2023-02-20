import torch
import torch.nn as nn
import torch.nn.functional as F



class parallelModel(nn.Module):
    def __init__(self, num_inputs, vocab_size, vocab_tags_size, emb_dim, hid_dim, max_length, num_layers, output_size, padding,
                 padding_tag, drop=0.4, bi=True):
        super(parallelModel, self).__init__()
        self.max_length = max_length
        lstm_out = 2 * hid_dim if bi else hid_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding)
        self.embedding_tag = nn.Embedding(vocab_tags_size, emb_dim, padding_idx=padding_tag)
        self.lstm = nn.LSTM(emb_dim, hid_dim, bidirectional=bi, num_layers=num_layers, dropout=drop, batch_first=True)
        self.lstm_tag = nn.LSTM(emb_dim, hid_dim, bidirectional=bi, num_layers=num_layers, dropout=drop,
                                batch_first=True)
        self.fc = nn.Linear(num_inputs * lstm_out, 148)
        self.fc2 = nn.Linear(148, output_size)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(drop)

    def forward(self, input_dict):
        x = input_dict['encodes']
        x_tag = input_dict['encodes_tag']
        out_emb_x = self.embedding(x)
        out_emb_x_tag = self.embedding_tag(x_tag)
        lstm_x, (hidden, cell_state) = self.lstm(out_emb_x)
        lstm_x_tag, (hidden_tag, cell_state_tag) = self.lstm(out_emb_x_tag)
        lstm_x = self.relu(lstm_x)
        lstm_x_tag = self.relu(lstm_x_tag)
        in_linear = torch.cat((lstm_x, lstm_x_tag), -1)
        print(lstm_x.shape)
        print(lstm_x_tag.shape)
        print(in_linear.shape)
        input()


        output1 = self.fc(in_linear.view(self.max_length, -1))
        output2 = self.fc2(self.relu(output1))
        output = F.log_softmax(output2, dim=1)

        return output


# class parallelModel2(nn.Module):
#     def __init__(self, num_inputs, vocab_size, vocab_tags_size, emb_dim, hid_dim, max_length, num_layers, output_size,
#                  padding,
#                  padding_tag, drop=0.4, bi=True):
#         super(parallelModel, self).__init__()
#         self.max_length = max_length
#         lstm_out = 2 * hid_dim if bi else hid_dim
#         self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding)
#         self.embedding_tag = nn.Embedding(vocab_tags_size, emb_dim, padding_idx=padding_tag)
#         self.lstm = nn.LSTM(emb_dim, hid_dim, bidirectional=bi, num_layers=num_layers, dropout=drop, batch_first=True)
#         # self.lstm_tag = nn.LSTM(emb_dim, hid_dim, bidirectional=bi, num_layers=num_layers, dropout=drop,
#         # batch_first=True)
#         self.fc = nn.Linear(lstm_out, output_size)
#         self.fc2 = nn.Linear(148, max_length)
#
#         self.relu = nn.ReLU()
#
#         self.dropout = nn.Dropout(drop)
#
#     def forward(self, input_dict):
#         x = input_dict['encodes']
#         x_tag = input_dict['encodes_tag']
#         out_emb_x = self.embedding(x)
#         out_emb_x_tag = self.embedding_tag(x_tag)
#         lstm_x, (hidden, cell_state) = self.lstm(out_emb_x)
#         # lstm_x_tag, (hidden_tag, cell_state_tag) = self.lstm(out_emb_x_tag)
#         lstm_x = self.relu(lstm_x)
#         # lstm_x_tag = self.relu(lstm_x_tag)
#         # in_linear = torch.cat((lstm_x, lstm_x_tag), -1)
#
#         output = self.fc(lstm_x.view(max_length, -1))
#
#         output = F.log_softmax(output, dim=1)
#
#         return output
