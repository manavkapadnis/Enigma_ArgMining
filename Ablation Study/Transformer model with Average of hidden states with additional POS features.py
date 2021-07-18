# -*- coding: utf-8 -*-
"""

    This python code implements argument key_point matching on EMNLP ArgMining 2021 Track 1 dataset
    Here, the outputs of the pretrained models are taken and the average of last 2 or last 3 hidden states is fed into the dense layers

"""

# Import the Required Libraries"""

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
warnings.filterwarnings('ignore')
import pickle
import nltk
import math
import os
import json
import random
import re
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import (DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, TensorDataset)
from pandarallel import pandarallel

# Initialization
pandarallel.initialize(progress_bar = True)

SEED = 0
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda'

path_dataset = '/dataset/'

# use the below two lines for storing model on noun features and comment the next two lined
path_predictions_folder = '/predictions/'
save_model_folder = '/model/'

max_len_arg = 55
max_len_kp = 32
max_len_topic = 12
max_len_sent_1_sts = 58
max_len_sent_2_sts = 65
median_len_sent_1_sts = 9
median_len_sent_2_sts = 9
max_noun_encoded_feature_len = 66
max_dependency_encoded_feature_len = 66
max_len_input = 128
model_with_no_token_types =['roberta', 'bart' ,'distilbert','deberta', 'xlmroberta', 'xlnet','xlnetlarge', 'robertalarge', 'bartlarge','debertalarge','xlmrobertalarge']

#Function to make TensorDataset of the files

def make_dataset(tokenizer, args,kps,topics,features, labels, max_len_input, model_with_no_token_types = model_with_no_token_types, model_name='roberta'):
    
    all_input_ids = []
    all_token_type_ids = []
    all_attention_masks = []
    all_labels = [] 
    all_features=[]
    
    for arg,kp,topic,feature,label in zip(args,kps,topics,features,labels) :

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
        topic = emoji_pattern.sub(r'',topic)

        if model_name in model_with_no_token_types:

          encoded_input = tokenizer(kp+arg+topic,max_length = max_len_input, padding='max_length')
          all_input_ids.append(encoded_input['input_ids'])
          all_attention_masks.append(encoded_input['attention_mask'])
          #all_token_type_ids.append(encoded_input['token_type_ids'])
          all_labels.append(label)
          all_features.append(feature)

        else :

          encoded_input = tokenizer(kp+arg+topic,max_length = max_len_input, padding='max_length')
          all_input_ids.append(encoded_input['input_ids'])
          all_attention_masks.append(encoded_input['attention_mask'])
          all_token_type_ids.append(encoded_input['token_type_ids'])
          all_labels.append(label)
          all_features.append(feature)
          
    if model_name in model_with_no_token_types:
      all_input_ids = torch.tensor(all_input_ids).squeeze()
      all_attention_masks = torch.tensor(all_attention_masks).squeeze()
      all_features = torch.tensor(all_features).squeeze()
      all_labels = torch.tensor(all_labels)
      
      dataset = TensorDataset(all_input_ids, all_attention_masks,all_features, all_labels)

    else :
      all_input_ids = torch.tensor(all_input_ids).squeeze()
      all_token_type_ids = torch.tensor(all_token_type_ids).squeeze()
      all_attention_masks = torch.tensor(all_attention_masks).squeeze()
      all_features = torch.tensor(all_features).squeeze()
      all_labels = torch.tensor(all_labels) 

      dataset = TensorDataset(all_input_ids,all_token_type_ids, all_attention_masks,all_features, all_labels)

    return dataset

# Loading the Dataset

df_train = pd.read_csv(path_dataset + 'train_tfidf.csv')
df_val = pd.read_csv(path_dataset + 'val_tfidf.csv')
df_test  = pd.read_csv(path_dataset + 'final_test.csv')

# Concatenate train and dev set for training the model for creating test dataset predictions
df_train = pd.concat([df_train,df_val])

def give_list(sent):
  res = ast.literal_eval(sent)
  return res

# Generate Noun features
df_train['encoded_noun_features']= df_train['encoded_noun_features'].parallel_apply(lambda x: give_list(x))
df_val['encoded_noun_features']= df_val['encoded_noun_features'].parallel_apply(lambda x: give_list(x))
df_test['encoded_noun_features']= df_test['encoded_noun_features'].parallel_apply(lambda x: give_list(x))


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

model_name = 'bart'
model_path = 'facebook/bart-base'

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Make dataset on noun features  

train_dataset = make_dataset(tokenizer, df_train['arg'], df_train['key_point'], df_train['topic'],df_train['encoded_noun_features'], df_train['label'], max_len_input, model_with_no_token_types, model_name=model_name)
val_dataset = make_dataset(tokenizer, df_val['arg'], df_val['key_point'], df_val['topic'],df_val['encoded_noun_features'], df_val['label'], max_len_input, model_with_no_token_types, model_name=model_name)
test_dataset = make_dataset(tokenizer, df_test['arg'], df_test['key_point'], df_test['topic'],df_test['encoded_noun_features'], df_test['stance'], max_len_input, model_with_no_token_types, model_name=model_name)

# MODEL ARCHITECTURE

# Use this Class only for Bert base and Bert large model

class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()
        
        #Instantiating Pre trained model object 
        self.model_layer = AutoModel.from_pretrained(model_path)
        
        #Layers
        # the first dense layer will have 834 neurons if base model is used and 
        # 1090 neurons if large model is used

        self.dense_layer_1 = nn.Linear(1090, 256)
        self.dropout = nn.Dropout(0.4)
        self.dense_layer_2 = nn.Linear(256, 128)
        self.dropout_2 = nn.Dropout(0.4) 
        self.cls_layer = nn.Linear(128, 1, bias = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input_ids, attention_masks, token_type_ids, features):

        last_hidden_states = self.model_layer(input_ids=input_ids, attention_mask=attention_masks,token_type_ids = token_type_ids).encoder_hidden_states
        
        # Took average of last three for other experiment
        pooled_output = (last_hidden_states[0] + last_hidden_states[1]) / 2 

        # Combining the noun features and pooler output
        concat = torch.cat((pooled_output,features), dim =1)
        
        x = self.dense_layer_1(concat)
        x = self.dropout(x)
        x_1 = self.dense_layer_2(x)
        x_2 = self.dropout_2(x_1)
        
        logits = self.cls_layer(x_2)
        output = self.sigmoid(logits)

        return output

# Use this Class for the rest of transformer models

class NonPoolerTransformer(nn.Module):

    def __init__(self):
        super(NonPoolerTransformer, self).__init__()
        
        #Instantiating Pre trained model object 
        self.model_layer = AutoModel.from_pretrained(model_path)
        
        #Layers
        # the first dense layer will have 834 neurons if base model is used and 
        # 1090 neurons if large model is used

        self.dense_layer_1 = nn.Linear(1090, 256)
        self.dropout = nn.Dropout(0.4)
        self.dense_layer_2 = nn.Linear(256, 128)
        self.dropout_2 = nn.Dropout(0.4)
        self.cls_layer = nn.Linear(128, 1, bias = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input_ids, attention_masks,features):

        
        last_hidden_states = self.model_layer(input_ids=input_ids, attention_mask=attention_masks,token_type_ids = token_type_ids).encoder_hidden_states
        
        # Took average of last three for other experiment
        pooled_output = (last_hidden_states[0] + last_hidden_states[1]) / 2

        ## Combining the noun features and model pooler output
        concat = torch.cat((pooled_output, features),dim =1)

        x = self.dense_layer_1(concat)
        x = self.dropout(x)
        x_1 = self.dense_layer_2(x)
        x_2 = self.dropout_2(x_1)
        
        logits = self.cls_layer(x_2)
        output = self.sigmoid(logits)

        return output

# Use Non Pooler Transformer as the experiment was done on bart and deberta
model = NonPoolerTransformer()

model = model.to(device)

BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 3
ACCUMULATION_STEPS = 2
DROPOUT = 0.4
gold_data_dir = path_dataset

PARAMS = {'model_name': model_name,'model_path': model_path,'lr': LEARNING_RATE, 'epoch_nr': EPOCHS, 'batch_size': BATCH_SIZE, 'accumulation_steps': ACCUMULATION_STEPS,'dropout': DROPOUT}

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
    #print("\nֿ** loading predictions:")
    arg_df = arg_df[["arg_id", "topic", "stance"]]
    predictions_df = load_predictions(predictions_file, kp_df["key_point_id"].unique())

    #make sure each arg_id has a prediction
    predictions_df = pd.merge(arg_df, predictions_df, how="left", on="arg_id")

    #handle arguements with no matching key point
    predictions_df["key_point_id"] = predictions_df["key_point_id"].fillna("dummy_id")
    predictions_df["score"] = predictions_df["score"].fillna(0)

    #merge each argument with the gold labels
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
                    #print(f"key point {invalid_kp} doesn't appear in the key points file and will be ignored")
                    invalid_keypoints.add(invalid_kp)
            if valid_kps:
                best_kp = max(valid_kps.items(), key=lambda x: x[1])
                arg.append(arg_id)
                kp.append(best_kp[0])
                scores.append(best_kp[1])
        #print(f"\tloaded predictions for {len(arg)} arguments")
        
        return pd.DataFrame({"arg_id" : arg, "key_point_id": kp, "score": scores})

def get_ap(df, label_column, top_percentile=0.5):
    top = int(len(df)*top_percentile)
    df = df.sort_values('score', ascending=False).head(top)
    # after selecting top percentile candidates, we set the score for the dummy kp to 1, to prevent it from increasing the precision.
    df.loc[df['key_point_id'] == "dummy_id", 'score'] = 0.99
    return average_precision_score(y_true=df[label_column], y_score=df["score"])

def calc_mean_average_precision(df, label_column):
    precisions = [get_ap(group, label_column) for _, group in df.groupby(["topic", "stance"])]
    return np.mean(precisions)

def evaluate_predictions(merged_df,name = 'train'):
    #print("\n** running evalution:")
    mAP_strict = calc_mean_average_precision(merged_df, "label_strict")
    mAP_relaxed = calc_mean_average_precision(merged_df, "label_relaxed")
                         
    print(f"mAP strict= {mAP_strict} ; mAP relaxed = {mAP_relaxed}")

# Train and Predict Functions

def evaluate_model(test_dataset,df, model,  model_name, mode = 'train'):
    
    save_predictions_name = model_name+ '__VAL_PREDS_'+ 'SEED_'+ str(SEED) + '_dense_layer' +'_epoc_'+ str(EPOCHS)+'_lr_'+ str(LEARNING_RATE)+'_b_s_'+ str(BATCH_SIZE) +'_accumulation_steps_'+ str(ACCUMULATION_STEPS) +'_input_type_kp_arg_topic_feature'

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
                b_input_ids, b_input_mask,b_features, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device),batch[3].to(device)
                ypred = model(b_input_ids, b_input_mask,b_features)
            else:
                b_input_ids,b_token_type, b_input_mask,b_features, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device),batch[4].to(device)
                ypred = model(b_input_ids, b_input_mask,b_token_type,b_features)
                
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

    return all_preds,true_labels, val_losses

def train_and_evaluate(train_dataset,df,model, filepath, model_name, batch_size = BATCH_SIZE, learning_rate = LEARNING_RATE, epochs = EPOCHS,accumulation_steps = ACCUMULATION_STEPS):
  
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
        b_input_ids, b_input_mask,b_features, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        outputs = model(b_input_ids, b_input_mask,b_features)
      else:
        b_input_ids,b_token_type, b_input_mask,b_features, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
        outputs = model(b_input_ids, b_input_mask,b_token_type,b_features)
            
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
    
  return save_model,train_losses, val_losses

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
        b_input_ids, b_input_mask,b_features, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        ypred = model(b_input_ids, b_input_mask,b_features)
      else:
        b_input_ids,b_token_type, b_input_mask,b_features, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device),batch[4].to(device)
        ypred = model(b_input_ids, b_input_mask,b_token_type,b_features)

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

def give_test_results(pred_file_path):
  print('The strict and relaxed scores on the test set predictions are: ')
  arg_df, kp_df, labels_df = load_kpm_data(gold_data_dir, subset="test")
  merged_df = get_predictions(pred_file_path, labels_df, arg_df, kp_df)
  evaluate_predictions(merged_df)

# Save the model, predictions and evaluate the model on the test set

save_model, train_losses, val_losses = train_and_evaluate(train_dataset,df_train, model, save_model_folder, model_name = 'bart', batch_size = BATCH_SIZE, learning_rate = LEARNING_RATE, epochs = EPOCHS, accumulation_steps = ACCUMULATION_STEPS)
test_preds_path = predict_model (test_dataset,df_test, save_model,model_name = 'bart')
give_test_results(test_preds_path)