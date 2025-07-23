import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import nltk
import gc
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

import os
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import csv
from collections import defaultdict
import copy
from transformers import BertTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq

import wandb
import random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


"""**DATA EXTRACTION**"""

df = pd.read_excel("Categorii Newsbox 10.xlsx", sheet_name=None, engine="openpyxl")
print(type(df))

df

dfs = []
for sheet_name, df_sheet in df.items():
  df_sheet["label"] = sheet_name
  dfs.append(df_sheet)

for sheet_name, df_sheet in df.items():
  print(df_sheet.head())

df_final = pd.concat(dfs, ignore_index=True) 
print(df_final['label'])

print(type(df_final))

df_final.head(190)

"""**DATA SPLIT IN 70% TRAIN 15% VALIDATE AND 15% TEST**"""

X = df_final.drop("label", axis=1)
Y = df_final["label"]
num_labels = len(set(Y))

print(X.shape)
print(Y.shape)


label_encoder = LabelEncoder()

text_columns = X.select_dtypes(include=['object']).columns
print(f"Text columns: {text_columns}")

X['combined_text'] = X[text_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

print(X[['combined_text']].head())

X_train, X_temp, Y_train, Y_temp = train_test_split(X["combined_text"], Y, test_size=0.3, random_state=42 )  

X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42) 

Y_train = label_encoder.fit_transform(Y_train)
Y_val = label_encoder.transform(Y_val)
Y_test = label_encoder.transform(Y_test)

print(len(Y))

print(f"train shape : {X_train.shape}")
print(f"val shape : {X_val.shape}")
print(f"test shape : {X_test.shape}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"first : num_labels {num_labels}")


print(type(X_train.iloc[0]))
print(type(X_val.iloc[0]))
print(type(X_test.iloc[0]))

print(set(Y_train))
print(set(Y_val))
print(set(Y_test))


"""## **BERT MODEL FOR ROMANIAN LANGUAGE** ##"""

def load_semi_supervised(lr_param):
  best_metric = -1
  best_metric_epoch = -1
  lr_param = lr_param

  model_name = "dumitrescustefan/bert-base-romanian-cased-v1"
  model_BERT = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
  tokenizer_BERT = BertTokenizer.from_pretrained(model_name)
  data_collator_BERT = DataCollatorWithPadding(tokenizer=tokenizer_BERT)

  optimizer = torch.optim.AdamW(model_BERT.parameters(), lr=lr_param, weight_decay=0.00001) 
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
  
  return tokenizer_BERT, data_collator_BERT, model_BERT, scheduler, optimizer, best_metric, best_metric_epoch 

def load_semi_supervised_mt0(lr_param):
    model_name = "bigscience/mt0-base"
    tokenizer_MT0 = AutoTokenizer.from_pretrained(model_name)
    model_MT0 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    data_collator_MT0 = DataCollatorForSeq2Seq(tokenizer=tokenizer_MT0, model=model_MT0)

    optimizer = torch.optim.AdamW(model_MT0.parameters(), lr=lr_param, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    return tokenizer_MT0, data_collator_MT0, model_MT0, scheduler, optimizer


"""**DATASET CLASS (LAZY INSTANTIATION)**"""

class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
      self.texts = list(texts) 
      self.labels = list(labels) 
      self.tokenizer = tokenizer 
      self.max_length = max_length 

    def __len__(self):
      return len(self.texts) 

    def __getitem__(self, idx):
      text = self.texts[idx] 
      label = self.labels[idx] 

      encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length, 
            return_tensors="pt"
        )


      return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long), 

        }
    
lr_param = 5e-6
num_workers = 4 
batch_size = 32
"""**DATASETS AND LOADERS**"""
tokenizer_BERT, data_collator_BERT, model_BERT, scheduler, optimizer, best_metric, best_metric_epoch = load_semi_supervised(lr_param)

dataset_train = CustomTextDataset(X_train, Y_train, tokenizer_BERT) 
dataset_test = CustomTextDataset(X_test, Y_test, tokenizer_BERT) 
dataset_val = CustomTextDataset(X_val, Y_val, tokenizer_BERT) 

test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=data_collator_BERT, num_workers=num_workers, persistent_workers=False) # loderul pentru testare impartim ca la antrenare doar ca nu vrem shuffle
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=data_collator_BERT, num_workers=num_workers, persistent_workers=False) # loaderul pentru antrenare impartim in batch-uri de cate 32
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=data_collator_BERT, num_workers=num_workers, persistent_workers=False) # loaderul pentru antrenare impartim in batch-uri de cate 32 si selectam shuffle ca sa asigura cross-entropy

labels = [batch["labels"].cpu().numpy() for batch in train_loader]
labels = np.concatenate(labels)

class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights) # aici era fara 

print(set(type(label) for label in Y_train))
print(set(type(label) for label in Y_val))
print(set(type(label) for label in Y_test))


"""**MT0 FUNCTION**"""

model_name = "bigscience/mt0-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

lr_param_mto = 5e-5 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
criterion = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_param_mto, weight_decay=0.01) 
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.to(device)

"""**LOADERS AND DATASETS FOR MT0**"""

class CustomTextDatasetMT0(Dataset):
    def __init__(self, texts, labels, tokenizer, label_map, max_length=512, label_max_length=16):
        self.texts = list(texts) 
        self.labels = list(labels) 
        self.tokenizer = tokenizer 
        self.label_map = label_map 
        self.max_length = max_length 
        self.label_max_length = label_max_length 

    def __len__(self):
        return len(self.texts) 

    def __getitem__(self, idx):
        text = self.texts[idx] 
        label_text = self.label_map[self.labels[idx]] 
        input_encoding = self.tokenizer( #
            text, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt" 
        )

        label_encoding = self.tokenizer( 
            label_text, 
            truncation=True, 
            max_length=self.label_max_length, 
            return_tensors="pt" 
        )
        labels = label_encoding["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0).cpu(),
            "attention_mask": input_encoding["attention_mask"].squeeze(0).cpu(),
            "labels": labels.cpu()
        }

label_map = {i: label for i, label in enumerate(label_encoder.classes_)} 
tokenizer_MT0, data_collator_MT0, model_MT0, scheduler, optimizer = load_semi_supervised_mt0(lr_param_mto)

dataset_train_mto = CustomTextDatasetMT0(X_train, Y_train, tokenizer_MT0, label_map) 
dataset_test_mto = CustomTextDatasetMT0(X_test, Y_test, tokenizer_MT0, label_map) 
dataset_val_mto = CustomTextDatasetMT0(X_val, Y_val, tokenizer_MT0, label_map) 

test_loader_mto = DataLoader(dataset_test_mto, batch_size=32, shuffle=False, collate_fn=data_collator_MT0) 
train_loader_mto = DataLoader(dataset_train_mto, batch_size=32, shuffle=True, collate_fn=data_collator_MT0) 
val_loader_mto = DataLoader(dataset_val_mto, batch_size=32, shuffle=False, collate_fn=data_collator_MT0) 

for batch in test_loader_mto:
    print(batch)
    break


def MT0():
  """## **MT0 model** ##"""

  """Aici in bucla de antrenare am detokenizat tokenii adica am transformat numere inapoi in stringuri ca sa compar cuvintele intre ele pentru ca:
  Prima oara am trecut etichetele printr-un label encoder adica le-am facut int-uri gen si dupa aceea le-am trecut printr-o clasa cu instantiere lazy care foloseste un tokenizer si gen aici am folosit un label_map care imi transforma inapoi din int in string si dupa ce sunt trecute prin clasa o sa am practic iarasi int-uri dar care sunt tokeni deci prin urmare am folosit un decodificator de la token ca sa transform numerele inapoi iarasi in string-uri ca sa le compar gen pentru ca MT0 foloseste eticheta ca cuvant propriu zis nu ca int.
  """

  model.to(device) 
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
  torch.cuda.empty_cache()
  wandb.login()

 
  lr = lr_param_mto
  epochs = 3

  best_metric = -1
  best_metric_epoch = -1
  train_accuracy_values = []
  val_accuracy_values = []

  run = wandb.init(
      project="supervised mt0",
      config={
          "learning_rate": lr,
          "epochs": epochs,
      },
      reinit=True,
      force=True
  )

  wandb.define_metric("epoch")
  wandb.define_metric("train_loss", step_metric="epoch")
  wandb.define_metric("train_accuracy", step_metric="epoch")
  wandb.define_metric("val_loss", step_metric="epoch")
  wandb.define_metric("val_accuracy", step_metric="epoch")
  wandb.define_metric("batch_loss", step_metric="epoch")


  offset = random.random() / 5
  print(f"lr: {lr}")


  for epoch in range(epochs):

      print("---------------")
      print(f"Epoch {epoch + 1}/{epochs}")

 

      model_MT0.train() 
      model_MT0.to(device)
      total_loss = 0 
      correct_train = 0
      total_train = 0

      for batch in train_loader_mto: 
          optimizer.zero_grad() 

          input_ids = batch["input_ids"].to(device) 
          attention_mask = batch["attention_mask"].to(device) 
          labels = batch["labels"].to(device)

          outputs = model_MT0(input_ids=input_ids, attention_mask=attention_mask, labels=labels) 
          loss = outputs.loss 

          loss.backward() 
          optimizer.step() 
          total_loss += loss.item() 

          preds = model_MT0.generate(input_ids, attention_mask=attention_mask) 

          decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
          decoded_labels = tokenizer.batch_decode(
              torch.where(labels == -100, tokenizer.pad_token_id, labels).cpu(),
              skip_special_tokens=True
          ) 
          correct_train += sum([pred.strip().lower() == label.strip().lower() for pred, label in zip(decoded_preds, decoded_labels)])
          total_train += len(decoded_preds)

          del outputs, loss
          torch.cuda.empty_cache()

      avg_loss_train = total_loss / len(train_loader_mto)
      train_accuracy = correct_train / total_train
      train_accuracy_values.append(train_accuracy)
      wandb.log({
        "train_loss": avg_loss_train,
        "train_accuracy": train_accuracy,
        "epoch": epoch
      })

      print(f"Epoch {epoch+1}, Training Loss: {avg_loss_train}, Training Accuracy: {train_accuracy}")



      model_MT0.eval()
      total_val_loss = 0
      correct_val = 0
      total_val = 0
      y_true = []
      y_pred = []

      with torch.no_grad():
        for batch in val_loader_mto: 
            input_ids = batch["input_ids"].to(device) 
            attention_mask = batch["attention_mask"].to(device) 
            labels = batch["labels"].to(device) 

            outputs = model_MT0(input_ids=input_ids, attention_mask=attention_mask, labels=labels) 
            loss = outputs.loss 
            total_val_loss += loss.item() 

 
            preds = model_MT0.generate(input_ids, attention_mask=attention_mask) 
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(
                torch.where(labels == -100, tokenizer.pad_token_id, labels).cpu(),
                skip_special_tokens=True
            )
            correct_val += sum([pred.strip().lower() == label.strip().lower() for pred, label in zip(decoded_preds, decoded_labels)])
            total_val += len(decoded_preds)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
      avg_loss_val = total_val_loss / len(val_loader_mto)
      val_accuracy = correct_val / total_val

      if val_accuracy > best_metric:
        best_metric = val_accuracy
        best_metric_epoch = epoch + 1
        torch.save(model_MT0.state_dict(), "best_model_mto_standard.pth")
        print(f"Better model saved at epoch {best_metric_epoch} with val_accuracy: {best_metric:.4f}")
      wandb.log({
        "val_loss": avg_loss_val,
        "val_accuracy": val_accuracy,
        "epoch": epoch
      })

      scheduler.step(avg_loss_val)
      print(f"Epoch {epoch+1}, Validation Loss: {avg_loss_val}, Validation Accuracy: {val_accuracy}")


  """**EVALUATING MT0**"""

  wandb.init()
  wandb.login()

  model_MT0.load_state_dict(torch.load("best_model_mto_standard.pth"))
  model_MT0.to(device)
  model_MT0.eval()
  y_true_test = []
  y_pred_test = []
  total_test_loss = 0
  correct_test = 0
  total_test = 0

  with torch.no_grad():
    for batch in test_loader_mto:
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels = batch["labels"].to(device)

      outputs =  model_MT0(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss
      total_test_loss += loss
      preds = model_MT0.generate(input_ids, attention_mask=attention_mask)

      decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
      decoded_labels = tokenizer.batch_decode(
          torch.where(labels == -100, tokenizer.pad_token_id, labels).cpu(),
          skip_special_tokens=True
      )
      correct_test += sum([pred.strip().lower() == label.strip().lower() for pred, label in zip(decoded_preds, decoded_labels)])

      total_test += len(decoded_preds)
      y_true_test.extend([label.strip().lower() for label in decoded_labels])
      y_pred_test.extend([preds.strip().lower() for preds in decoded_preds])


  avg_loss_test = total_test_loss / len(test_loader_mto)
  test_accuracy = correct_test / total_test
  wandb.log({"test_loss": avg_loss_test, "test_accuracy": test_accuracy})

  print(f" Test Loss: {avg_loss_test:.4f}, Test Accuracy: {test_accuracy:.4f}")

  print("Classification Report on Test Set:")
  print(classification_report(y_true_test, y_pred_test, zero_division=0))

  wandb.finish()



df_1mil = df = pd.read_json("dump.jsonl", lines=True)
print(type(df_1mil))

df_1mil.head(10)
label_map = {i: label for i, label in enumerate(label_encoder.classes_)} 
print(f"num_labels : {num_labels}")
class CustomTextDatasetSemi_Supervised(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        combined_text = f"{row['title']} {row['text']}"

        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

unlabeled_dataset = CustomTextDatasetSemi_Supervised(df_1mil, tokenizer_MT0)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False, collate_fn=data_collator_MT0, num_workers=num_workers, persistent_workers=True) # 32
unlabeled_loader_size = len(unlabeled_loader)
print(f"unlabeled_loader_size {unlabeled_loader_size}")

def ensure_tensor(example):
    for key in ["input_ids", "attention_mask", "labels"]:
        if not isinstance(example[key], torch.Tensor):
            example[key] = torch.tensor(example[key], dtype=torch.long).cpu()
        else:
            example[key] = example[key].cpu()

for batch in train_loader_mto:
  print(batch)
  print(f"type batch : {type(batch)}")
  break

lr = lr_param_mto


best_metric = -1
best_metric_epoch = -1
train_accuracy_values = []
val_accuracy_values = []
predicted = defaultdict(list)

lambda_u = 1.0
max_samples = 1000
max_samples_loader = 1000

num_iterations = 3 
epochs = 10 
confidence_threshold = 0.65
optimizer = torch.optim.AdamW(model_MT0.parameters(), lr=lr_param_mto, weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
criterion = torch.nn.CrossEntropyLoss()
class_names = [label_map[i] for i in range(num_labels)]

wandb.login()
api = wandb.Api()
user_email = api.viewer.email
print(f"[DEBUG] Logged in as: {user_email}")


for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")
    print("---------------")

    if wandb.run is not None:
        wandb.finish()

    run = wandb.init(
        project="semi-supervised MT0 standard",
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
        name=f"iteration - {iteration + 1}",
        reinit=True,
        force=True
    )


    if wandb.run is not None:                                                                                                   
        print(f"[DEBUG] Run name: {wandb.run.name}")                                                                            
        print(f"[DEBUG] Run id: {wandb.run.id}")                                                                                
        print(f"[DEBUG] Project: {wandb.run.project}")                                                                      
    else:
        print("[DEBUG] No active wandb run!")
    
    offset = random.random() / 5
    
    print(f"lr: {lr}")

    wandb.define_metric("epoch", step_metric=None)
    wandb.define_metric("iteration", step_metric=None)

    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("train_accuracy", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("val_accuracy", step_metric="epoch")
    wandb.define_metric("test_accuracy", step_metric="epoch")
    wandb.define_metric("test_loss", step_metric="epoch")
    wandb.define_metric("batch_loss", step_metric="epoch")
    wandb.define_metric("f1-score", step_metric="iteration")
    wandb.define_metric("precision-score", step_metric="iteration")
    wandb.define_metric("recall-score", step_metric="iteration")

    tokenizer_MT0, data_collator_MT0, model_MT0, scheduler, optimizer = load_semi_supervised_mt0(lr_param_mto)
    full_unlabeled_dataset = list(unlabeled_loader.dataset)

    for epoch in range(epochs):

      print("---------------")
      print(f"Epoch {epoch + 1}/{epochs}")


      model_MT0.train()
      model_MT0.to(device)
      total_loss = 0 
      correct_train = 0
      total_train = 0

      for batch in train_loader_mto: 
          optimizer.zero_grad() 

          input_ids = batch["input_ids"].to(device) 
          attention_mask = batch["attention_mask"].to(device) 
          labels = batch["labels"].to(device) 

          outputs = model_MT0(input_ids=input_ids, attention_mask=attention_mask, labels=labels) 
          loss = outputs.loss

          loss.backward() 
          optimizer.step() 
          total_loss += loss.item() 

          preds = model_MT0.generate(input_ids, attention_mask=attention_mask) 

          decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
          decoded_labels = tokenizer.batch_decode(
              torch.where(labels == -100, tokenizer.pad_token_id, labels).cpu(),
              skip_special_tokens=True
          )
          correct_train += sum([pred.strip().lower() == label.strip().lower() for pred, label in zip(decoded_preds, decoded_labels)])
          total_train += len(decoded_preds)

          del outputs, loss
          torch.cuda.empty_cache()

      avg_loss_train = total_loss / len(train_loader_mto)
      train_accuracy = correct_train / total_train
      train_accuracy_values.append(train_accuracy)
      wandb.log({
        "train_loss": avg_loss_train,
        "train_accuracy": train_accuracy,
        "epoch": epoch
      })

      print(f"Epoch {epoch+1}, Training Loss: {avg_loss_train}, Training Accuracy: {train_accuracy}")



      model_MT0.eval()
      total_val_loss = 0
      correct_val = 0
      total_val = 0
      y_true = []
      y_pred = []

      with torch.no_grad():
        for batch in val_loader_mto: 
            input_ids = batch["input_ids"].to(device) 
            attention_mask = batch["attention_mask"].to(device) 
            labels = batch["labels"].to(device) 

            outputs = model_MT0(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss 
            total_val_loss += loss.item() 
            preds = model_MT0.generate(input_ids, attention_mask=attention_mask) 
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(
                torch.where(labels == -100, tokenizer.pad_token_id, labels).cpu(),
                skip_special_tokens=True
            )
            correct_val += sum([pred.strip().lower() == label.strip().lower() for pred, label in zip(decoded_preds, decoded_labels)])
            total_val += len(decoded_preds)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
      avg_loss_val = total_val_loss / len(val_loader_mto)
      val_accuracy = correct_val / total_val

      if val_accuracy > best_metric:
        best_metric = val_accuracy
        best_metric_epoch = epoch + 1
        torch.save(model_MT0.state_dict(), "best_model_mto_standard.pth")
        print(f"Better model saved at epoch {best_metric_epoch} with val_accuracy: {best_metric:.4f}")
      wandb.log({
        "val_loss": avg_loss_val,
        "val_accuracy": val_accuracy,
        "epoch": epoch
      })

      scheduler.step(avg_loss_val)
      print(f"Epoch {epoch+1}, Validation Loss: {avg_loss_val}, Validation Accuracy: {val_accuracy}")

    """**EVALUATING MT0**"""

    model_MT0.load_state_dict(torch.load("best_model_mto_standard.pth"))
    model_MT0.to(device)
    model_MT0.eval()
    y_true_test = []
    y_pred_test = []
    total_test_loss = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for batch in test_loader_mto:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs =  model_MT0(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_test_loss += loss
            preds = model_MT0.generate(input_ids, attention_mask=attention_mask) 
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(
                torch.where(labels == -100, tokenizer.pad_token_id, labels).cpu(),
                skip_special_tokens=True
            )
            correct_test += sum([pred.strip().lower() == label.strip().lower() for pred, label in zip(decoded_preds, decoded_labels)])

            total_test += len(decoded_preds)
            y_true_test.extend([label.strip().lower() for label in decoded_labels])
            y_pred_test.extend([preds.strip().lower() for preds in decoded_preds])


    avg_loss_test = total_test_loss / len(test_loader_mto)
    test_accuracy = correct_test / total_test
    wandb.log({"test_loss": avg_loss_test, "test_accuracy": test_accuracy})
    
    f1_value = f1_score(y_true=y_true_test, y_pred=y_pred_test, average="macro")
    precision_value = precision_score(y_true=y_true_test, y_pred=y_pred_test, average="macro", zero_division=0)
    recall_value = recall_score(y_true=y_true_test, y_pred=y_pred_test, average="macro", zero_division=0)

    wandb.log({
        "f1-score": f1_value,
        "precision-score": precision_value,
        "recall-score": recall_value,
        "iteration": int(iteration)
    })

    print(f" Test Loss: {avg_loss_test:.4f}, Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report on Test Set:")
    print(classification_report(y_true_test, y_pred_test, zero_division=0))

    cm_standard_MT0 = confusion_matrix(y_true=y_true_test, y_pred=y_pred_test)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm_standard_MT0, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix MT0 STANDARD')

    plt.savefig('confusion_matrix_mt0_standard.png', dpi=300, bbox_inches='tight')


    pseudo_labeled = []
    unlabeled = []
    unlabeled_text = []
    labeled_text = []

    steps_per_iteration = 20

    model_MT0.eval()
    with torch.no_grad():

        for i, batch in enumerate(unlabeled_loader):
            print(f"batch {i + 1} / {len(unlabeled_loader)}")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if i >= steps_per_iteration:
                break 

            batch_size = batch["input_ids"].shape[0]
            for i in range(batch_size): 
                print(f"text {i + 1} / {batch_size}")
                input_id_x = batch["input_ids"][i].unsqueeze(0).to(device)
                attention_mask_x = batch["attention_mask"][i].unsqueeze(0).to(device)

                tokenized_labels = []
                scores = []

                for label_text in label_map.values():
                   tokenized_label = tokenizer_MT0(
                      label_text,
                      padding=True,
                      truncation=True,
                      max_length=16,
                      return_tensors="pt"
                   )["input_ids"].to(device)
                   tokenized_labels.append(tokenized_label)
                
                for tokenized_label in tokenized_labels:
                    decoder_input_ids = tokenized_label[:, :-1].to(device)
                    targets = tokenized_label[:, 1:].to(device)

                    with torch.no_grad():
                        outputs = model_MT0(
                            input_ids=input_id_x,
                            attention_mask=attention_mask_x,
                            decoder_input_ids=decoder_input_ids,
                            return_dict=True
                        )

                        logits = outputs.logits
                        log_probs = F.log_softmax(logits, dim=-1)
                        token_log_probs = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)
                        score = token_log_probs.mean()
                        scores.append(score.item())

                    del outputs
                    gc.collect()
                    torch.cuda.empty_cache()

                torch_scores = torch.tensor(scores)
                probs = F.softmax(torch_scores, dim=0)

                best_index = torch.argmax(probs).item()
                best_label_id = list(label_map.keys())[best_index]
                best_label_text = label_map[best_label_id]
                confidence = probs[best_index].item()
                print(f"confidence is : {confidence}")

                if confidence >= confidence_threshold:
                   pseudo_labeled.append({
                            "input_ids": input_id_x.cpu().squeeze(0),
                            "attention_mask": attention_mask_x.cpu().squeeze(0), 
                            "labels": tokenized_labels[best_index].squeeze(0).cpu(),
                            "label_id": best_label_id,
                            "confidence": confidence
                        })
                else:
                    unlabeled.append({
                            "input_ids": input_id_x.cpu().squeeze(0),
                            "attention_mask": attention_mask_x.cpu().squeeze(0)
                        })

                
        print(f" Am adăugat {len(pseudo_labeled)} pseudo-etichetări.")
        if len(pseudo_labeled) == 0:
            print("Nicio etichetă suficient de sigură. Ne oprim.")
            break
        
        class_counts = defaultdict(list)

        for ex in pseudo_labeled:
          label = ex["label_id"]
          class_counts[label].append(ex)

        final_pseudo = []
        
        for label, examples in class_counts.items():
          sorted_examples = sorted(examples, key=lambda x: x["confidence"], reverse=True)
          print(f"pentru clasa {label_map[label]} am scos {len(sorted_examples)}")
          final_pseudo.extend(sorted_examples[:(3 * len(sorted_examples) // 4)])
        
          predicted[label].extend(copy.deepcopy(examples))
        class_counts_deepcopy = copy.deepcopy(class_counts)
        
        for example in final_pseudo:
          if "confidence" in example:
            del example["confidence"]
          if "label_id" in example:
            del example["label_id"]
        
          ensure_tensor(example)

        pseudo_labeled = final_pseudo

        seen = set()

        for ex in pseudo_labeled: 
            key = (
                tuple(ex["input_ids"].tolist()),
                tuple(ex["attention_mask"].tolist())
            )
            seen.add(key)
        print(f"[DEBUG] seen there are : {len(seen)}")

        unlabeled = [
            ex for ex in full_unlabeled_dataset
            if (
                tuple(ex["input_ids"].tolist()),
                tuple(ex["attention_mask"].tolist())
            ) not in seen
        ]

        print(f"[DEBUG] unlabeled there are : {len(unlabeled)}")

        train_data = list(train_loader_mto.dataset)
        train_data.extend(pseudo_labeled)

        train_loader_mto = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=data_collator_MT0, num_workers=num_workers, persistent_workers=False) # TRUE sau FALSE

        for i, batch in enumerate(train_loader_mto):
          assert "labels" in batch, f"Missing 'labels' in batch {i}"
          assert isinstance(batch["labels"], torch.Tensor), f"'labels' not a tensor in batch {i}"
          break

        unlabeled_loader = DataLoader(unlabeled, batch_size=32, shuffle=False, collate_fn=data_collator_MT0, num_workers=num_workers, persistent_workers=True)
        unlabeled_loader_size = len(unlabeled_loader)

        if confidence_threshold >= 0.55:
            confidence_threshold -= 0.05
        
    wandb.finish()




max_displayed = 10

with open("predicted_top10_texts_semi_supervised_Standard_SSL_MT0.csv", mode = 'w', newline="", encoding="utf-8") as file:
  writer=csv.writer(file)
  writer.writerow(["label"] + [f"text_{i+1}" for i in range(max_displayed)] + [f"conf_{i+1}" for i in range(max_displayed)])

  for label, examples in predicted.items():
    sorted_examples = sorted(examples, key=lambda x: x["confidence"], reverse=True)
    top_10 = sorted_examples[:max_displayed]

    texts = [tokenizer_MT0.decode(ex["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True) for ex in top_10]
    confidences = [f"{ex['confidence']:.4f}" for ex in top_10]

    texts += [""] * (max_displayed - len(texts))
    confidences += [""] * (max_displayed - len(confidences))

    writer.writerow([label_map[label]] + texts + confidences)


print("\n Am generat csv-ul !!! \n")
