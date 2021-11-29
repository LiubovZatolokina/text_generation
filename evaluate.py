import datasets
import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2Config, BertForNextSentencePrediction
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline

bert_model = BertForNextSentencePrediction.from_pretrained('bert_model_nsp')

config_gpt2 = GPT2Config.from_json_file('gpt2_model/config.json')
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2_model/pytorch_model.bin', config=config_gpt2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def evaluate_model(model, eval_dataloader, tokenizer, model_name):
    metric_bleu = datasets.load_metric('bleu')
    metric_rouge = datasets.load_metric('rouge')

    for inputs, labels in tqdm(eval_dataloader):
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        output_lst = [generator(i, max_length=10, num_return_sequences=1)[0]['generated_text'].split() for i in inputs]
        label_lst = [i.split() for i in labels]
        metric_bleu.add_batch(predictions=np.array(output_lst)[np.newaxis, ...],
                              references=np.array(label_lst)[np.newaxis, ...])
        metric_rouge.add_batch(predictions=np.array(output_lst)[np.newaxis, ...],
                               references=np.array(label_lst)[np.newaxis, ...])

    final_bleu = metric_bleu.compute()
    final_rouge = metric_rouge.compute()

    print(f'{model_name} BLEU: ', final_bleu)
    print(f'{model_name} ROUGE: ', final_rouge)
