import os
from tqdm import tqdm
import torch
import json

class TrainHuge:
    def __init__(self, model, epochs, loss, optim, exp_num, results, device, cv=False):
        self.model = model
        self.epochs = epochs
        self.loss = loss
        self.optim = optim
        self.exp_num = exp_num
        self.results = results
        self.device = device
        self.cv = cv

    def train_model(self, train_loader, valid_loader, train_tag_loader, valid_tag_loader):
        train_epoch_losses = list()
        train_epoch_acc = list()
        valid_epoch_losses = list()
        valid_epoch_acc = list()
        for epoch in range(self.epochs):
            self.model.train()
            epoch_tr_loss = 0
            epoch_tr_acc = 0

            tqdm_desc = f'Epoch: {epoch+1}/{self.epochs} => epoch loss: {epoch_tr_loss}; epoch acc: {epoch_tr_acc}'
            iterator = tqdm(train_loader, tqdm_desc, len(train_loader), leave=True)
            for train_batch, train_tag_batch in zip(iterator, train_tag_loader):
                train_data = train_batch['data'].to(self.device)
                train_tag_data = train_tag_batch['data'].to(self.device)
                train_labels = train_batch['labels'].to(self.device)
                input_data = {'encodes': train_data, 'encodes_tag': train_tag_data}
                self.optim.zero_grad()
                outputs = self.model(input_data)
                # print(outputs.float())
                loss = self.loss(outputs.float(), train_labels)
                loss.backward()
                self.optim.step()
                epoch_tr_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                epoch_tr_acc += (predicted == train_labels).sum().item()
                acc_percentage = epoch_tr_acc / len(train_loader.dataset)
                tqdm_desc = f'Epoch: {epoch + 1}/{self.epochs} => epoch loss: {epoch_tr_loss/len(train_loader):.4f}; epoch acc: {acc_percentage:.4f}'
                iterator.set_description(tqdm_desc)


            avg_tr_loss = epoch_tr_loss/len(train_loader)
            avg_tr_acc = epoch_tr_acc/len(train_loader.dataset)
            avg_val_loss, avg_val_acc = self.evaluate(valid_loader, valid_tag_loader)
            print(f'Epoch: {epoch + 1}/{self.epochs} => epoch valid loss: {avg_val_loss :.4f}; epoch valid acc: {avg_val_acc:.4f}')
            train_epoch_losses.append(avg_tr_loss)
            train_epoch_acc.append(avg_tr_acc)
            valid_epoch_losses.append(avg_val_loss)
            valid_epoch_acc.append(avg_val_acc)
            self.save_results(train_epoch_losses, valid_epoch_losses,
                              train_epoch_acc, valid_epoch_acc, epoch)

    def evaluate(self, valid_loader, valid_tag_loader):
        correct = 0
        total = 0
        losses = 0
        with torch.no_grad():
            self.model.eval()
            for each_valid, each_valid_tag in zip(valid_loader, valid_tag_loader):
                valid_data = each_valid['data'].to(self.device)
                valid_labels = each_valid['labels'].to(self.device)
                valid_tag_data = each_valid_tag['data'].to(self.device)

                valid_input_data = {'encodes': valid_data, 'encodes_tag': valid_tag_data}
                outputs = self.model(valid_input_data)
                loss = self.loss(outputs, valid_labels)
                _, predicted = torch.max(outputs.data, 1)
                total += valid_labels.size(0)
                correct += (predicted == valid_labels).sum().item()
                losses += loss.item()
        avg_valid_loss = losses / len(valid_loader)
        avg_valid_acc = correct / len(valid_loader.dataset)
        return avg_valid_loss, avg_valid_acc

    def save_results(self, train_losses, valid_losses, train_acc, valid_acc, epoch):
        folder_name = 'results_and_params'
        model_folder = f'model_params_{self.exp_num}' if not self.cv else f'model_params_cv_comb_{self.exp_num}'
        folder = os.path.join(self.results, folder_name)
        model_folder = os.path.join(folder, model_folder)
        results_dict = {
            'train_loss': train_losses,
            'train_acc': train_acc,
            'valid_loss': valid_losses,
            'valid_acc': valid_acc
        }
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        with open(os.path.join(self.results, 'results.json'), 'w') as file_results:
            json.dump(results_dict, file_results)

        model_name = os.path.join(model_folder, f'model_{self.exp_num}_{epoch}')
        torch.save(self.model, model_name)