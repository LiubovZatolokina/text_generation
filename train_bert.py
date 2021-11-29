import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizer, BertForNextSentencePrediction

from data_preparation import prepare_data_for_training

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
train_loader, test_loader = prepare_data_for_training()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
num_epochs = 300
model_saving_path = './bert_model.pt'


def train_model(model, dataloaders, optimizer, num_epochs, model_saving_path):
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    since = time.time()
    tb = SummaryWriter()
    for epoch in tqdm(range(num_epochs)):
        loss_dict = {}
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    encoding = tokenizer(np.array(inputs).tolist(), np.array(labels).tolist(),
                                         return_tensors='pt', padding=True)
                    encoding = encoding.to(device)
                    labels = torch.tensor([1] * encoding["input_ids"].size(0)).unsqueeze(0)
                    loss = model(**encoding, labels=labels.to(device)).loss
                    running_loss += loss.item() * len(inputs[0])
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            loss_dict[phase] = epoch_loss
            model.save_pretrained("bert_model_nsp")
            torch.save(model.state_dict(), model_saving_path)

        tb.add_scalars('Loss: epoch', {'Train': loss_dict['train'], 'Valid': loss_dict['valid']}, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    torch.cuda.empty_cache()
    bert_model.to(device)
    optimizer = torch.optim.Adam(bert_model.parameters(), lr=learning_rate)
    dataloaders_dict = {'train': train_loader, 'valid': test_loader}
    model_ft = train_model(bert_model, dataloaders_dict, optimizer, num_epochs, model_saving_path)