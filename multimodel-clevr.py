# %%
from sklearn.model_selection import train_test_split
import pandas as pd

# %%
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel
import torchvision.transforms as T
import torch
from torch import nn

# %%

exp_id = 3

# %%
train_df = pd.read_pickle("data/CLEVR_v1.0/train-data.pkl")

# %%
val_df = pd.read_pickle("data/CLEVR_v1.0/val-data.pkl")

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Using: ", device)


# %%
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_encoder = AutoModel.from_pretrained("bert-base-uncased")
for p in text_encoder.parameters():
    p.requires_grad = False

# %%
image_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
image_encoder = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")

for p in image_encoder.parameters():
    p.requires_grad = False

# %%
if torch.cuda.device_count() > 1:
    print("Available GPU's: ", torch.cuda.device_count())
    image_encoder = nn.DataParallel(image_encoder)
    text_encoder = nn.DataParallel(text_encoder)


image_encoder.to(device)
text_encoder.to(device)


# %%
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

class EasyQADataset(Dataset):

    def __init__(self,df,
                 image_encoder, 
                 text_encoder,
                 image_processor, 
                 tokenizer,
              ):
        self.df = df
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = self.df['image'][idx]
        question = self.df['question'][idx] 
        label = self.df['answer_code'][idx]

        image_inputs = self.image_processor(image, return_tensors="pt")
        image_inputs = {k:v.to(device) for k,v in image_inputs.items()}
        image_outputs = self.image_encoder(**image_inputs)
        image_embedding = image_outputs.pooler_output
        image_embedding = image_embedding.view(-1)
        image_embedding = image_embedding.detach()

        text_inputs = self.tokenizer(question, return_tensors="pt")
        text_inputs = {k:v.to(device) for k,v in text_inputs.items()}
        text_outputs = self.text_encoder(**text_inputs)
        # text_embedding = text_outputs.pooler_output
        text_embedding = text_outputs.last_hidden_state[:,0,:]
        text_embedding = text_embedding.view(-1)
        text_embedding = text_embedding.detach()

        encoding={}
        encoding["image_emb"] = image_embedding
        encoding["text_emb"] = text_embedding
        encoding["label"] = torch.tensor(label)

        return encoding

# %%
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

train_dataset = EasyQADataset(
                           df=train_df,
                           image_encoder = image_encoder,
                           text_encoder = text_encoder,
                           tokenizer = text_tokenizer,
                           image_processor = image_processor,
                           )

eval_dataset = EasyQADataset(
                           df=val_df,
                           image_encoder = image_encoder,
                           text_encoder = text_encoder,
                           tokenizer = text_tokenizer,
                           image_processor = image_processor,
                          )

# %%
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 128
eval_batch_size = 32
dataloader_train = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset), 
                              batch_size=batch_size)
dataloader_validation = DataLoader(eval_dataset, 
                                   sampler=SequentialSampler(eval_dataset), 
                                   batch_size=eval_batch_size)

# %%
from sklearn.metrics import accuracy_score

def accuracy_score_func(preds, labels):
    return accuracy_score(labels, preds)

# %%
import random
from torch import nn
from tqdm.notebook import tqdm
import numpy as np
import requests

criterion = nn.CrossEntropyLoss()

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals, confidence = [], [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch.values())
        
        inputs = {'image_emb':  batch[0],'text_emb': batch[1]}  

        with torch.no_grad():        
            outputs = model(**inputs)
            
        labels =  batch[2]  
        loss = criterion(outputs.view(-1, 28), labels.view(-1))
        loss_val_total += loss.item()

        probs   = torch.max(outputs.softmax(dim=1), dim=-1)[0].detach().cpu().numpy()
        outputs = outputs.argmax(-1)
        logits = outputs.detach().cpu().numpy()
        label_ids = labels.cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
        confidence.append(probs)
    
    loss_val_avg = loss_val_total/len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    confidence = np.concatenate(confidence, axis=0)
            
    return loss_val_avg, predictions, true_vals, confidence

def train():

  train_history = open(f"exp{exp_id}_models/train_history.csv", "w")
  log_hdr  = "Epoch, train_loss, train_acc, val_loss, val_acc"
  train_history.write(log_hdr  + "\n")
  train_f1s = []
  val_f1s = []
  train_losses = []
  val_losses = []
  min_val_loss = -1
  max_auc_score = 0
  epochs_no_improve = 0
  early_stopping_epoch = 3
  early_stop = False

  for epoch in tqdm(range(1, epochs+1)):

      model.train()
      loss_train_total = 0
      train_predictions, train_true_vals = [], []

      progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

      for batch in progress_bar:
          model.zero_grad()
          batch = tuple(b.to(device) for b in batch.values())

          inputs = {'image_emb':  batch[0],'text_emb': batch[1]} 
          labels =  batch[2]

          outputs = model(**inputs)
          loss = criterion(outputs.view(-1, 28), labels.view(-1))
          loss_train_total += loss.item()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          
          logits = outputs.argmax(-1)
          logits = logits.detach().cpu().numpy()
          label_ids = labels.cpu().numpy()
          train_predictions.append(logits)
          train_true_vals.append(label_ids)

          optimizer.step()
          scheduler.step()
          
          progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
          
          
      
      train_predictions = np.concatenate(train_predictions, axis=0)
      train_true_vals = np.concatenate(train_true_vals, axis=0)

      tqdm.write(f'\nEpoch {epoch}')
      loss_train_avg = loss_train_total/len(dataloader_train)            
      tqdm.write(f'Training loss: {loss_train_avg}')
      train_f1 = accuracy_score_func(train_predictions, train_true_vals)
      tqdm.write(f'Train Acc: {train_f1}')
      
      val_loss, predictions, true_vals,_ = evaluate(dataloader_validation)
      val_f1 = accuracy_score_func(predictions, true_vals)
      tqdm.write(f'Validation loss: {val_loss}')
      tqdm.write(f'Val Acc: {val_f1}')

      if val_f1 >= max_auc_score:
          tqdm.write('\nSaving best model')
          torch.save(model.state_dict(), f'exp{exp_id}_models/epoch_{epoch}.model')          
          max_auc_score = val_f1

      train_losses.append(loss_train_avg)
      val_losses.append(val_loss)
      train_f1s.append(train_f1)
      val_f1s.append(val_f1)
      log_str  = "{}, {}, {}, {}, {}".format(epoch, loss_train_avg, train_f1, val_loss, val_f1)
      train_history.write(log_str + "\n")

      if min_val_loss < 0:
          min_val_loss = val_loss
      else:
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_epoch:
                early_stop = True
                break
            else:
                continue    


  if early_stop:
    print("Early Stopping activated at epoch -", epoch )
    print("Use the checkpoint at epoch - ", epoch - early_stopping_epoch)

  train_history.close()
  return train_losses, val_losses

# %%
class BaseNetwork(nn.Module):

    def __init__(self, hyperparms=None):

        super(BaseNetwork, self).__init__()        
        self.dropout = nn.Dropout(0.3)
        self.vision_projection = nn.Linear(768, 768) 
        self.text_projection = nn.Linear(768, 768)
        self.fc1 = nn.Linear(768, 256) 
        self.bn1 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, 28) 
        W = torch.Tensor(768, 768)
        self.W = nn.Parameter(W)
        self.relu_f = nn.ReLU()
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        
    def forward(self, image_emb, text_emb):

        x1 = image_emb   
        x1 = torch.nn.functional.normalize(x1, p=2, dim=1)
        Xv = self.relu_f(self.vision_projection(x1))
        
        x2 = text_emb
        x2 = torch.nn.functional.normalize(x2, p=2, dim=1)
        Xt = self.relu_f(self.text_projection(x2))

        Xvt = Xv * Xt
        Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))

        Xvt = self.fc1(Xvt)
        Xvt = self.bn1(Xvt)
        Xvt = self.dropout(Xvt)
        Xvt = self.classifier(Xvt)

        return Xvt
    

# %%

class NewQAEarlyFusionNetwork(nn.Module):

    def __init__(self, hyperparms=None):

        super(NewQAEarlyFusionNetwork, self).__init__()        
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, 28) 
        W = torch.Tensor(768, 768)
        self.W = nn.Parameter(W)
        self.relu_f = nn.ReLU()
        # initialize weight matrices
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        
    def forward(self, image_emb, text_emb):

        x_img = image_emb   
        
        x_text = text_emb

        Xvt = x_img * x_text
        Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))


        Xvt = self.fc1(Xvt)
        Xvt = self.bn1(Xvt)
        Xvt = self.relu_f(Xvt)
        Xvt = self.fc2(Xvt)
        Xvt = self.bn2(Xvt)
        Xvt = self.relu_f(Xvt)
        Xvt = self.dropout(Xvt)
        Xvt = self.classifier(Xvt)

        return Xvt
    
# %%
torch.cuda.empty_cache()

import math

model = NewQAEarlyFusionNetwork()

if torch.cuda.device_count() > 1:
    print("Available GPU's: ", torch.cuda.device_count())
    model = nn.DataParallel(model)

model.to(device)

# %%
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

optimizer = AdamW(model.parameters(),
                  lr=5e-5, 
                  weight_decay = 1e-5,
                  eps=1e-8
                  )
                  
epochs = 30
train_steps=20000
print("train_steps", train_steps)
warm_steps = train_steps * 0.1
print("warm_steps", warm_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=warm_steps,
                                            num_training_steps=train_steps)

# %%
from matplotlib import pyplot as plt

train_losses, val_losses =  train()
torch.cuda.empty_cache()
plt.plot(train_losses)
plt.plot(val_losses)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(f"exp{exp_id}_loss_plot.png")

