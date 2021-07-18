
"""
    This python code implements Boosting algorithm on the EMNLP ArgMining 2021 Track 1 Dataset
    5 base models are used for boosting
    The input to the pretrained model is the combination of key point, argument and topic

"""


# Import the required Libraries

import numpy as np
import pandas as pd
import regex as re
import random as rn
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,average_precision_score, precision_score,precision_recall_curve
from tqdm.notebook import tqdm
from tqdm import trange
import warnings
import pickle
import nltk
import math
import os
import json
import random
from random import choices
import re
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import (DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, TensorDataset)
from pandarallel import pandarallel

# Initialisation of pandarallel
pandarallel.initialize(progress_bar = True)

# Definition of Seed
# Seeds 0, 1 and 2 were tried for the experiments 

SEED = 0        
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda'

# Definition of directory path

path_dataset = '/dataset/'
path_predictions_folder = '/predictions/'
save_model_folder = '/models/'

max_len_arg = 55
max_len_kp = 32
max_len_topic = 12
max_len_sent_1_sts = 58
max_len_sent_2_sts = 65
median_len_sent_1_sts = 9
median_len_sent_2_sts = 9
max_len_input = 128
model_with_no_token_types =['roberta', 'bart' ,'distilbert','deberta', 'xlmroberta', 'xlnet','xlnetlarge', 'robertalarge', 'bartlarge','debertalarge','xlmrobertalarge']

# Function to make TensorDataset of the files

def make_dataset(tokenizer, args, kps, topics, labels, max_len_input, model_with_no_token_types = model_with_no_token_types, model_name = 'roberta'):
    
    all_input_ids = []
    all_token_type_ids = []
    all_attention_masks = []
    all_labels = [] 
    
    for arg, kp, topic, label in zip(args, kps, topics, labels) :

        arg = re.sub('[^a-zA-Z]', ' ', arg)
        kp = re.sub('[^a-zA-Z]', ' ', kp)
        topic = re.sub('[^a-zA-Z]', ' ', topic)

        url = re.compile(r'https?://\S+|www\.\S+')
        arg = url.sub(r'',arg)
        kp = url.sub(r'',kp)
        topic = url.sub(r'',topic)
        
        html=re.compile(r'<.*?>')
        arg = html.sub(r'',arg)
        kp = html.sub(r'',kp)
        topic = html.sub(r'',topic)

        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        
        arg = emoji_pattern.sub(r'',arg)
        kp = emoji_pattern.sub(r'',kp)

        if model_name in model_with_no_token_types:

          encoded_input = tokenizer(kp + arg + topic, max_length = max_len_input, padding = 'max_length')
          all_input_ids.append(encoded_input['input_ids'])
          all_attention_masks.append(encoded_input['attention_mask'])
          all_labels.append(label)

        else :

          encoded_input = tokenizer(kp + arg + topic, max_length = max_len_input, padding = 'max_length')
          all_input_ids.append(encoded_input['input_ids'])
          all_attention_masks.append(encoded_input['attention_mask'])
          all_token_type_ids.append(encoded_input['token_type_ids'])
          all_labels.append(label)
          
    if model_name in model_with_no_token_types:
      all_input_ids = torch.tensor(all_input_ids).squeeze()
      all_attention_masks = torch.tensor(all_attention_masks).squeeze()
      all_labels = torch.tensor(all_labels)
      
      dataset = TensorDataset(all_input_ids, all_attention_masks, all_labels)

    else :
      all_input_ids = torch.tensor(all_input_ids).squeeze()
      all_token_type_ids = torch.tensor(all_token_type_ids).squeeze()
      all_attention_masks = torch.tensor(all_attention_masks).squeeze()
      all_labels = torch.tensor(all_labels) 

      dataset = TensorDataset(all_input_ids,all_token_type_ids, all_attention_masks, all_labels)

    return dataset

# Loading the Dataset

df_train = pd.read_csv(path_dataset + 'train_tfidf.csv')
df_val = pd.read_csv(path_dataset + 'val_tfidf.csv')
df_test  = pd.read_csv(path_dataset + 'final_test.csv')

# Concatenate the train and validation set for final training
df_train = pd.concat([df_train,df_val])

# Reset the index, necessary for boosting
df_train = df_train.reset_index(drop = True)

"""
    Types_of_models = model : tokenizer model_path

 - 'bert':  'bert-base-uncased'
 - 'roberta':  'roberta-base'
 - 'bart':  "facebook/bart-base"
 - 'distilbert': 'distilbert-base-uncased'
 - 'deberta': 'microsoft/deberta-base'
 - 'debertalarge': 'microsoft/deberta-large'
 - 'xlnet' : 'xlnet-base-cased'
 - 'xlnetlarge' : 'xlnet-large-cased'
 - 'xlmrobertalarge' : 'xlm-roberta-large'
 - 'bartlarge' : 'facebook/bart-large'
 - 'bertlarge':  'bert-large-uncased'
 - 'robertalarge':  'roberta-large'

"""

model_name = 'bartlarge'
model_path = 'facebook/bart-large'

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Train, validation and Test dataset compatible with pytorch framework

train_dataset = make_dataset(tokenizer, df_train['arg'], df_train['key_point'], df_train['topic'], df_train['label'], max_len_input, model_with_no_token_types, model_name=model_name)
val_dataset = make_dataset(tokenizer, df_val['arg'], df_val['key_point'], df_val['topic'], df_val['label'], max_len_input, model_with_no_token_types, model_name=model_name)
test_dataset = make_dataset(tokenizer, df_test['arg'], df_test['key_point'], df_test['topic'], df_test['stance'], max_len_input, model_with_no_token_types, model_name=model_name)

# MODEL ARCHITECTURE

# transformer class for bert-base and bert-large model only

class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()
        
        # Instantiating Pre trained model object 
        self.model_layer = AutoModel.from_pretrained(model_path)
        
        # Layers
        # The first dense layer will have 768 neurons if base model is used and 
        # 1024 neurons if large model is used

        self.dense_layer_1 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.4)
        self.dense_layer_2 = nn.Linear(256, 128)
        self.dropout_2 = nn.Dropout(0.4) 
        self.cls_layer = nn.Linear(128, 1, bias = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input_ids, attention_masks, token_type_ids):

        pooled_output = self.model_layer(input_ids=input_ids, attention_mask=attention_masks,token_type_ids = token_type_ids).pooler_output
        
        x = self.dense_layer_1(pooled_output)
        x = self.dropout(x)
        x_1 = self.dense_layer_2(x)
        x_2 = self.dropout_2(x_1)
        
        logits = self.cls_layer(x_2)
        output = self.sigmoid(logits)

        return output

# Non Pooler Transformer Class for models other than bert-base and bert-large

class NonPoolerTransformer(nn.Module):

    def __init__(self):
        super(NonPoolerTransformer, self).__init__()
        
        # Instantiating Pre trained model object 
        self.model_layer = AutoModel.from_pretrained(model_path)

        # Layers
        # The first dense layer will have 768 if base model is used and 
        # 1024 if large model is used

        self.dense_layer_1 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.4)
        self.dense_layer_2 = nn.Linear(256, 128)
        self.dropout_2 = nn.Dropout(0.2)
        self.cls_layer = nn.Linear(128, 1, bias = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input_ids, attention_masks):

        hidden_state = self.model_layer(input_ids=input_ids, attention_mask=attention_masks)[0]
        pooled_output = hidden_state[:, 0]

        x = self.dense_layer_1(pooled_output)
        x = self.dropout(x)
        x_1 = self.dense_layer_2(x)
        x_2 = self.dropout_2(x_1)

        logits = self.cls_layer(x_2)
        output = self.sigmoid(logits)

        return output

# Set the hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 3
ACCUMULATION_STEPS = 2
DROPOUT = 0.4
gold_data_dir = path_dataset

# Evaluation functions

def load_kpm_data(gold_data_dir, subset):
    
    arguments_file = os.path.join(gold_data_dir, f"arguments_{subset}.csv")
    key_points_file = os.path.join(gold_data_dir, f"key_points_{subset}.csv")
    labels_file = os.path.join(gold_data_dir, f"labels_{subset}.csv")

    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)
    
    return arguments_df, key_points_df, labels_file_df

def get_predictions(predictions_file, labels_df, arg_df, kp_df):
    
    arg_df = arg_df[["arg_id", "topic", "stance"]]
    predictions_df = load_predictions(predictions_file, kp_df["key_point_id"].unique())

    # Make sure each arg_id has a prediction
    predictions_df = pd.merge(arg_df, predictions_df, how="left", on="arg_id")

    # Handle arguements with no matching key point
    predictions_df["key_point_id"] = predictions_df["key_point_id"].fillna("dummy_id")
    predictions_df["score"] = predictions_df["score"].fillna(0)

    # Merge each argument with the gold labels
    merged_df = pd.merge(predictions_df, labels_df, how="left", on=["arg_id", "key_point_id"])

    merged_df.loc[merged_df['key_point_id'] == "dummy_id", 'label'] = 0
    merged_df["label_strict"] = merged_df["label"].fillna(0)
    merged_df["label_relaxed"] = merged_df["label"].fillna(1)

    return merged_df

def load_predictions(predictions_dir, correct_kp_list):

    arg =[]
    kp = []
    scores = []
    invalid_keypoints = set()

    with open(predictions_dir, "r") as f_in:
    
        res = json.load(f_in)
    
        for arg_id, kps in res.items():
    
            valid_kps = {key: value for key, value in kps.items() if key in correct_kp_list}
            invalid = {key: value for key, value in kps.items() if key not in correct_kp_list}
    
            for invalid_kp, _ in invalid.items():
    
                if invalid_kp not in invalid_keypoints:
                    invalid_keypoints.add(invalid_kp)
    
            if valid_kps:
                best_kp = max(valid_kps.items(), key=lambda x: x[1])
                arg.append(arg_id)
                kp.append(best_kp[0])
                scores.append(best_kp[1])

        return pd.DataFrame({"arg_id" : arg, "key_point_id": kp, "score": scores})

def get_ap(df, label_column, top_percentile=0.5):

    top = int(len(df)*top_percentile)
    df = df.sort_values('score', ascending=False).head(top)
    
    # After selecting top percentile candidates, we set the score for the dummy kp to 1, to prevent it from increasing the precision.
    df.loc[df['key_point_id'] == "dummy_id", 'score'] = 0.99
    return average_precision_score(y_true=df[label_column], y_score=df["score"])

def calc_mean_average_precision(df, label_column):

    precisions = [get_ap(group, label_column) for _, group in df.groupby(["topic", "stance"])]
    return np.mean(precisions)

def evaluate_predictions(merged_df, name = 'train'):

    mAP_strict = calc_mean_average_precision(merged_df, "label_strict")
    mAP_relaxed = calc_mean_average_precision(merged_df, "label_relaxed")
                         
    print(f"mAP strict= {mAP_strict} ; mAP relaxed = {mAP_relaxed}")

# Train and Predict Functions

def evaluate_model(test_dataset, df, model,  model_name, mode = 'train'):
    
    save_predictions_name = model_name+ '__VAL_PREDS_'+ 'SEED_'+ str(SEED) + '_dense_layer' +'_epoc_'+ str(EPOCHS)+'_lr_'+ str(LEARNING_RATE)+'_b_s_'+ str(BATCH_SIZE) +'_accumulation_steps_'+ str(ACCUMULATION_STEPS) +'_input_type_kp_arg_topic_'

    y_preds = []
    val_losses = []
    criterion = nn.BCELoss()
    list_of_batch_losses = []
    
    if mode in ['train','val']:
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    with torch.no_grad():
        acc_epoch = []
        
        epoch_iterator = tqdm(test_dataloader, desc="Iteration")
        
        for step, batch in enumerate(epoch_iterator):
            
            model.eval()
            
            if model_name in model_with_no_token_types:
                b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                ypred = model(b_input_ids, b_input_mask)
            else:
                b_input_ids,b_token_type, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                ypred = model(b_input_ids, b_input_mask,b_token_type)
                
            b_labels_copy = torch.reshape(b_labels, (b_labels.shape[0], 1))
            loss_batch = criterion(ypred, b_labels_copy.float())
            list_of_batch_losses.append(loss_batch.detach().cpu().numpy())
            
            ypred = ypred.cpu().numpy()
            b_labels = batch[-1].cpu().detach().numpy()
        
            ypred = np.hstack(ypred)
            y_preds.append(ypred)
    
    epoch_loss = np.mean(list_of_batch_losses)
    val_losses.append(epoch_loss)
    
    args = df['arg_id']
    kps = df['key_point_id']
    true_labels = df['label']
    topics = df['topic']
    stances = df['stance']
    all_preds = []

    for i in tqdm(range(len(y_preds))):
      for p in y_preds[i]:
        all_preds.append(p)
            
    print('Val evaluation....')
    
    pred_file = pd.DataFrame({"arg_id" : args, "key_point_id": kps, "score": all_preds})
    args = {}
    kps = {}

    for arg,kp,score in zip(pred_file['arg_id'],pred_file['key_point_id'],pred_file['score']):
        args[arg] = {}

    for arg,kp,score in zip(pred_file['arg_id'],pred_file['key_point_id'],pred_file['score']):
        args[arg][kp] = score

    with open(path_predictions_folder + save_predictions_name + '_' + 'predictions.p.', 'w') as fp:
        fp.write(json.dumps(args))
        fp.close()
    
    arg_df, kp_df, labels_df = load_kpm_data(path_dataset, subset="dev")
    merged_df = get_predictions(path_predictions_folder + save_predictions_name + '_' + 'predictions.p.', labels_df, arg_df, kp_df)
    
    evaluate_predictions(merged_df,name = 'val')

    return all_preds, true_labels, val_losses

def train_and_evaluate(train_dataset, df, model, filepath, model_name, batch_size = BATCH_SIZE, learning_rate = LEARNING_RATE, epochs = EPOCHS,accumulation_steps = ACCUMULATION_STEPS):
  
  train_losses = []
  val_losses = []
    
  save_model = model_name+ '_SEED_'+ str(SEED) +'_dense_layer' +'_epoc_'+ str(epochs)+'_lr_'+ str(learning_rate)+'_b_s_'+ str(batch_size ) +'_accumulation_steps_'+ str(accumulation_steps) +'_input_type_kp_arg_topic' 
  save_predictions_name  = model_name+ '__TRAIN_PREDS_'+ 'SEED_'+ str(SEED) + '_dense_layer' +'_epoc_'+ str(epochs)+'_lr_'+ str(learning_rate)+'_b_s_'+ str(batch_size ) +'_accumulation_steps_'+ str(accumulation_steps) +'_input_type_kp_arg_topic'

  training_dataloader = DataLoader(train_dataset, batch_size )
  total_steps = len(training_dataloader) * epochs
  no_decay = ['bias', 'LayerNorm.weight']
  
  optimizer_grouped_parameters = [
                                  {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                  {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                                  ]

  optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps = 1e-8)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

  criterion = nn.BCELoss()
    
  model.zero_grad()

  for epoch_i in tqdm(range(epochs)):
    y_preds = []
    y_val = []
    list_of_batch_losses = []
    epoch_iterator = tqdm(training_dataloader, desc="Iteration")
    model.train()
    
    for step, batch in enumerate(epoch_iterator):
      if model_name in model_with_no_token_types:
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        outputs = model(b_input_ids, b_input_mask)
      else:
        b_input_ids,b_token_type, b_input_mask,b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        outputs = model(b_input_ids, b_input_mask,b_token_type)
            
      b_labels = torch.reshape(b_labels, (b_labels.shape[0], 1))
      loss = criterion(outputs, b_labels.float())
             
      list_of_batch_losses.append(loss.detach().cpu().numpy())
      
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
      ypred = outputs.detach().cpu().numpy()
      b_labels = batch[-1].cpu().detach().numpy()
      ypred = np.hstack(ypred)
      y_preds.append(ypred)

      if (step+1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        model.zero_grad()


    epoch_loss = np.mean(list_of_batch_losses)
    train_losses.append(epoch_loss)
    
    args = df['arg_id']
    kps = df['key_point_id']
    true_labels = df['label']
    topics = df['topic']
    stances = df['stance']
    all_preds = []
    
    for i in tqdm(range(len(y_preds))):
      for p in y_preds[i]:
        all_preds.append(p)

    print('Train evaluation....')
    
    pred_file = pd.DataFrame({"arg_id" : args, "key_point_id": kps, "score": all_preds})
    args = {}
    kps = {}

    for arg,kp,score in zip(pred_file['arg_id'],pred_file['key_point_id'],pred_file['score']):
        args[arg] = {}

    for arg,kp,score in zip(pred_file['arg_id'],pred_file['key_point_id'],pred_file['score']):
        args[arg][kp] = score

    with open(path_predictions_folder + save_predictions_name + '_' + 'predictions.p.', 'w') as fp:
        fp.write(json.dumps(args))
        fp.close()
    
    arg_df, kp_df, labels_df = load_kpm_data(path_dataset, subset="train")
    merged_df = get_predictions(path_predictions_folder + save_predictions_name + '_' + 'predictions.p.', labels_df, arg_df, kp_df)
    
    evaluate_predictions(merged_df,name = 'train')
    
    _,_, val_epoch_loss = evaluate_model(val_dataset,df_val, model,  model_name, mode = 'val')
    val_losses.append(val_epoch_loss)
    
  torch.save(model, save_model_folder +save_model+'.pt')

  print("Model is saved as : ",save_model)
  print("Use this to load the model")
    
  return save_model, train_losses, val_losses

def predict_model(test_dataset,df, save_model,model_name):
  preds = []

  test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
  save_predictions_name = "TEST_PREDS_"+ '_SEED_'+ str(SEED) + save_model

  model=torch.load(save_model_folder + save_model +'.pt')

  with torch.no_grad():
    
    epoch_iterator = tqdm(test_dataloader, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):
      model.eval()

      if model_name in model_with_no_token_types:
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        ypred = model(b_input_ids, b_input_mask)
      else:
        b_input_ids,b_token_type, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        ypred = model(b_input_ids, b_input_mask,b_token_type)

      ypred = ypred.cpu().numpy()
      ypred = np.hstack(ypred)

      preds.append(ypred)

  args = df['arg_id']
  kps = df['key_point_id']
  all_preds = []

  for i in tqdm(range(len(preds))):
    for p in preds[i]:
      all_preds.append(p)

  pred_file = pd.DataFrame({"arg_id" : args, "key_point_id": kps, "score": all_preds})

  args = {}
  kps = {}

  for arg,kp,score in zip(pred_file['arg_id'],pred_file['key_point_id'],pred_file['score']):
    args[arg] = {}

  for arg,kp,score in zip(pred_file['arg_id'],pred_file['key_point_id'],pred_file['score']):
    args[arg][kp] = score

  with open(path_predictions_folder + save_predictions_name + '_' + 'predictions.p.', 'w') as fp:
    fp.write(json.dumps(args))
    fp.close()

  print("The predictions are stored in the file : "+ path_predictions_folder  + save_predictions_name + '_' + 'predictions.p.')
  
  return path_predictions_folder + save_predictions_name + '_' + 'predictions.p.'

def return_accuracy(test_dataset, save_model,  model_name, mode = 'train'):
  y_preds = []
  y_val = []

  if mode in ['train','val']:
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    model=torch.load(save_model_folder +save_model+'.pt')
    

    with torch.no_grad():
      acc_epoch = []
      epoch_iterator = tqdm(test_dataloader, desc="Iteration")
      for step, batch in enumerate(epoch_iterator):
        model.eval()

        if model_name in model_with_no_token_types:
          b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
          ypred = model(b_input_ids, b_input_mask)

        else:
          b_input_ids,b_token_type, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
          ypred = model(b_input_ids, b_input_mask)

        ypred = ypred.cpu().numpy()
        b_labels = batch[-1].cpu().detach().numpy()
        
        ypred = np.hstack(ypred)

        y_preds.append(ypred)
        y_val.append(b_labels)

        ypred = (ypred > 0.5)

        acc_batch = accuracy_score(b_labels,ypred)
        acc_epoch.append(acc_batch)

    accuracy = np.mean(np.array(acc_epoch))
  print("Accuracy: {0}".format(np.mean(np.array(acc_epoch))))

  return y_preds, y_val, accuracy

# Function implementing the Boosting algorithm

def Boosting():

    # Probability distribution over the training set for sampling the data points
    D = np.ones(len(df_train)) / (len(df_train))

    # Data indices list used to sample indices of data points
    data_index = [i+1 for i in range(len(df_train['arg']))]

    # Store weights associated with each model
    alpha = np.zeros(5)

    # Iterate for 5 base model
    # Base model 1 considers the whole dataset 
    # Other models take 12000 data points for training

    for i in range(5):

        rn.seed(321)
        np.random.seed(321)
        torch.manual_seed(321)
        torch.cuda.manual_seed(321)
        device = 'cuda'

        sampled_args = []
        sampled_key_points = []
        sampled_topics = []
        sampled_labels = []
        sampled_indices = []
        
        if i == 0:

            train_dataset = make_dataset(tokenizer, df_train['arg'], df_train['key_point'], df_train['topic'], df_train['label'], max_len_input, model_with_no_token_types, model_name=model_name)

        else:

            sampled_indices = choices(data_index, D, k = 10000)

            for j in range(10000):
    
                sampled_args.append(df_train['arg'][sampled_indices[j] - 1])
                sampled_key_points.append(df_train['key_point'][sampled_indices[j] - 1])
                sampled_topics.append(df_train['topic'][sampled_indices[j] - 1])
                sampled_labels.append(df_train['label'][sampled_indices[j] - 1])
            
            temp_dict = {'arg': sampled_args, 'key_point': sampled_key_points, 'topic': sampled_topics, 'label': sampled_labels}
            temp_df = pd.DataFrame(temp_dict)

            train_dataset = make_dataset(tokenizer, temp_df['arg'], temp_df['key_point'], temp_df['topic'], temp_df['label'], max_len_input, model_with_no_token_types, model_name=model_name)            

        model = NonPoolerTransformer()
        model = model.to(device)

        save_model, train_losses, val_losses = train_and_evaluate(train_dataset, model, save_model_folder, model_name = 'bartlarge', batch_size = BATCH_SIZE, learning_rate = LEARNING_RATE, epochs = EPOCHS, accumulation_steps = 8, i = i)
        
        loss_list_bart = []
        for j in train_losses:
            loss_list_bart.append(float(j.detach().cpu().numpy()))

        loss_every_100 = []

        for j in range(len(loss_list_bart)):
            if (j%100==0):
                loss_every_100.append(loss_list_bart[j])

        plt.plot(loss_every_100[4:])

        y_pred, y_val, accuracy = return_accuracy(train_dataset, save_model,  model_name = 'bartlarge', mode = 'train')
        all_preds, true_labels, val_losses = evaluate_model(train_dataset, save_model,  model_name = 'bartlarge', mode = 'train')
      
        if accuracy < 0.5:
            break
        
        # Update the weight for model
        alpha[i] = 0.5 * np.log((accuracy / (1-accuracy)))

        thr = 0.5

        # Update D

        if i == 0:

            for row_pred, row_val in zip(y_pred, y_val):
                
                for cell_pred, cell_val in zip(row_pred, row_val):

                    if (cell_pred >= thr and cell_val == 1) or (cell_pred < thr and cell_val == 0):
                        D[j] = D[j] * np.exp(-alpha[i])
                    else:
                        D[j] = D[j] * np.exp(alpha[i])

            D = D / np.sum(D)

        else:
            
            j = 0
            for row_pred, row_val in zip(y_pred, y_val):

                for cell_pred, cell_val in zip(row_pred, row_val):
            
                    if (cell_pred >= thr and cell_val == 1) or (cell_pred < thr and cell_val == 0):
                        D[sampled_indices[j]-1] = D[sampled_indices[j]-1] * np.exp(-alpha[i])
                    else:
                        D[sampled_indices[j]-1] = D[sampled_indices[j]-1] * np.exp(alpha[i])
                
                j = j + 1
                
            D = D / np.sum(D)

        predict_model (test_dataset, df_test, save_model, model_name = 'bartlarge')

        del model
        del train_dataset
        torch.cuda.empty_cache()

    return alpha

alpha = Boosting()

# Set the predictions path depending on the model used

pred1_path = ''
pred2_path = ''
pred3_path = ''
pred4_path = ''
pred5_path = ''

# Load the predictions to be used for final prediction on test set

pred1 = load_predictions(pred1_path)
pred2 = load_predictions(pred2_path)
pred3 = load_predictions(pred3_path)
pred4 = load_predictions(pred4_path)
pred5 = load_predictions(pred5_path)

arg = []
kp = []
score = []

for i in range(len(pred1)):
    
    arg.append(pred1['arg_id'][i])
    kp.append(pred1['key_point_id'][i])    
    temp_score = alpha[0]*pred1['score'][i] + alpha[1]*pred2['score'][i] + alpha[2]*pred3['score'][i] + alpha[3]*pred4['score'][i] + alpha[4]*pred5['score'][i]
    score.append(temp_score)

result_df = pd.DataFrame({"arg_id" : arg, "key_point_id": kp, "score": score}) 

# Save the final test predictions

args = {}
kps = {}

for arg, kp, score in zip(result_df['arg_id'], result_df['key_point_id'], result_df['score']):
    args[arg] = {}

for arg, kp, score in zip(result_df['arg_id'], result_df['key_point_id'], result_df['score']):
    args[arg][kp] = score

with open('/predictions.p.', 'w') as fp:
      fp.write(json.dumps(args))
      fp.close() 

# Set the pred_file_path as per the model used

pred_file_path = ''

def give_test_results(pred_file_path):

  print('The strict and relaxed scores on the test set predictions are: ')
  arg_df, kp_df, labels_df = load_kpm_data(gold_data_dir, subset="test")
  merged_df = get_predictions(pred_file_path, labels_df, arg_df, kp_df)
  evaluate_predictions(merged_df)

# Give the score on test set
give_test_results(pred_file_path)