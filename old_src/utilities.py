import torch

def pretty_printer(data, extra_info=None, number_of_element=5):
    print(f'{extra_info} has length: {len(data)}')
    num_elements = number_of_element if number_of_element < len(data) else len(data)
    for idx, each_data in enumerate(data):
        print(f'{idx}: \t{each_data}')
        if idx == number_of_element:
            break


def vocabulary_printer(vocab, extra_info=None, number_of_element=10):
    num_elements = number_of_element if number_of_element < len(vocab) else len(vocab)
    print(f'{extra_info} has {len(vocab)} elements')
    for count, (word, index) in enumerate(vocab.items()):
        if count == number_of_element:
            break
        print(f'{word}:\t{index}')

    print(f"paddingindex:\t{vocab['<pad>']}")
    if '<unk>' in vocab.keys():
        print(f"out of vocabulary index:\t{vocab['<unk>']}")
    else:
        print(f'out of vocabulary index:\t{extra_info} does not include <unk> token')
    print('<<<=====================================>>>')


def show_correspondences(sentences, enc, tag, lab, set_type):
    # idx = random.randrange(len(enc))
    idx = 24
    print(f'Corresponding sentence to index of {idx} in {set_type} dataset:')
    print(f'Sentence: \t{sentences[idx]}')
    print(f'Encoded : \t{enc[idx]}')
    print(f'Tagged  : \t{tag[idx]}')
    print(f'Labelled: \t{lab[idx]}')
    print('<<<<<<=========================================================================>>>>>>')


def compute_precision(model, test_loader, test_tag_loader, pad_idx, device):
    all_predictions = list()
    all_labels = list()
    for indexed_elem, indexed_elem_tag in zip(test_loader, test_tag_loader):

        indexed_input = indexed_elem["data"].to(device)
        indexed_input_tag = indexed_elem_tag['data'].to(device)
        indexed_labels = indexed_elem["labels"].to(device)
        input_dict = {'encodes': indexed_input, 'encodes_tag': indexed_input_tag}
        predictions = model(input_dict)
        predictions = torch.argmax(predictions, -1).view(-1)
        labels = indexed_labels.view(-1)
        valid_indices = labels != pad_idx

        valid_predictions = predictions[valid_indices]
        valid_labels = labels[valid_indices]

        all_predictions.extend(valid_predictions.tolist())
        all_labels.extend(valid_labels.tolist())
    return all_predictions, all_labels