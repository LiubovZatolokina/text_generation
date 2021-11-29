import time

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from data_preparation import prepare_data_for_training

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

train_loader, test_loader = prepare_data_for_training()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

learning_rate = 1e-4
num_epochs = 300
model_saving_path = './gpt2_model.pt'


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
                    input_ = tokenizer(np.array(inputs).tolist(), return_tensors="pt", padding=True)
                    input_ = input_.to(device)
                    labels_ = tokenizer(np.array(labels).tolist(), return_tensors="pt", padding=True,
                                        max_length=input_['input_ids'].size()[1])
                    labels_ = labels_.to(device)
                    if input_['input_ids'].size()[1] > labels_['input_ids'].size()[1]:
                        labels_ = tokenizer(np.array(labels).tolist(), return_tensors="pt", padding='max_length',
                                            max_length=input_['input_ids'].size()[1])

                    if input_['input_ids'].size()[1] < labels_['input_ids'].size()[1]:
                        input_ = tokenizer(np.array(labels).tolist(), return_tensors="pt", padding='max_length',
                                            max_length=labels_['input_ids'].size()[1])
                    input_ = input_.to(device)
                    labels_ = labels_.to(device)
                    loss = model(input_['input_ids'], labels=labels_['input_ids']).loss
                    running_loss += loss.item() * len(inputs[0])
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(epoch_loss)
            loss_dict[phase] = epoch_loss
            model.save_pretrained("gpt2_model")
            torch.save(model.state_dict(), model_saving_path)

        tb.add_scalars('Loss: epoch', {'Train': loss_dict['train'], 'Valid': loss_dict['valid']}, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    torch.cuda.empty_cache()
    gpt2_model.to(device)
    optimizer = torch.optim.Adam(gpt2_model.parameters(), lr=learning_rate)
    dataloaders_dict = {'train': train_loader, 'valid': test_loader}
    model_ft = train_model(gpt2_model, dataloaders_dict, optimizer, num_epochs, model_saving_path)