import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
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

model_name = "dumitrescustefan/bert-base-romanian-cased-v1"
model_BERT = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels) 
tokenizer = BertTokenizer.from_pretrained(model_name) 
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
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
            "label": torch.tensor(label, dtype=torch.long),
        }
    
lr_param = 5e-6   
"""**DATASETS AND LOADERS**"""
tokenizer_BERT, data_collator_BERT, model_BERT, scheduler, optimizer, best_metric, best_metric_epoch = load_semi_supervised(lr_param)

dataset_train = CustomTextDataset(X_train, Y_train, tokenizer_BERT) 
dataset_test = CustomTextDataset(X_test, Y_test, tokenizer_BERT) 
dataset_val = CustomTextDataset(X_val, Y_val, tokenizer_BERT) 

test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False, collate_fn=data_collator_BERT) 
train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=data_collator_BERT) 
val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False, collate_fn=data_collator_BERT) 

labels = [batch["labels"].cpu().numpy() for batch in train_loader]
labels = np.concatenate(labels)

class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

print(set(type(label) for label in Y_train))
print(set(type(label) for label in Y_val))
print(set(type(label) for label in Y_test))


def BERT():
  """**Loss function and optimizer**"""
  optimizer = torch.optim.AdamW(model_BERT.parameters(), lr=lr_param, weight_decay=0.00001) 

  for batch in train_loader:
      print(batch)
      break

  print(f"train_loader labels : {len(train_loader.dataset.labels)}")
  print(f"val_loader labels : {len(val_loader.dataset.labels)}")

  """**TRAINING BERT**"""

  model_BERT.to(device) 
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

  wandb.login()

  lr = 0.001
  lr = lr_param
  epochs = 15

  best_metric = -1
  best_metric_epoch = -1
  train_accuracy_values = []
  val_accuracy_values = []

  run = wandb.init(
      project="supervised bert",
      config={
          "learning_rate": lr,
          "epochs": epochs,
      },
  )

  offset = random.random() / 5
  print(f"lr: {lr}")

  wandb.define_metric("epoch")
  wandb.define_metric("train_loss", step_metric="epoch")
  wandb.define_metric("train_accuracy", step_metric="epoch")
  wandb.define_metric("val_loss", step_metric="epoch")
  wandb.define_metric("val_accuracy", step_metric="epoch")
  wandb.define_metric("batch_loss", step_metric="epoch")

  for epoch in range(epochs):

      print("---------------")
      print(f"Epoch {epoch + 1}/{epochs}")



      model_BERT.train()
      total_loss = 0 
      correct_train = 0
      total_train = 0

      for batch in train_loader:
          optimizer.zero_grad() 

          input_ids = batch["input_ids"].to(device) 
          attention_mask = batch["attention_mask"].to(device) 
          labels = batch["labels"].to(device) 

          outputs = model_BERT(input_ids, attention_mask=attention_mask) 
          loss = criterion(outputs.logits, labels) 

          loss.backward() 
          optimizer.step()

          total_loss += loss.item() 
          preds = torch.argmax(outputs.logits, dim=1)
          correct_train += (preds == labels).sum().item()
          total_train += labels.size(0)
          wandb.log({"batch_loss": loss.item()})
          del outputs, loss
          torch.cuda.empty_cache()

      avg_loss_train = total_loss / len(train_loader)
      train_accuracy = correct_train / total_train
      train_accuracy_values.append(train_accuracy)
      wandb.log({
      "train_loss": avg_loss_train,
      "train_accuracy": train_accuracy,
      "epoch": epoch
      })

      print(f"Epoch {epoch+1}, Training Loss: {avg_loss_train}, Training Accuracy: {train_accuracy}")

      model_BERT.eval()
      total_val_loss = 0
      correct_val = 0
      total_val = 0
      y_true = []
      y_pred = []

      with torch.no_grad():
        for batch in val_loader: 
            input_ids = batch["input_ids"].to(device) 
            attention_mask = batch["attention_mask"].to(device) 
            labels = batch["labels"].to(device) 

            outputs = model_BERT(input_ids, attention_mask=attention_mask) 
            loss = criterion(outputs.logits, labels) 
            total_val_loss += loss.item() 

            preds = torch.argmax(outputs.logits, dim=1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            wandb.log({"batch_loss": loss.item()})

      avg_loss_val = total_val_loss / len(val_loader)
      val_accuracy = correct_val / total_val

      if val_accuracy > best_metric:
        best_metric = val_accuracy
        best_metric_epoch = epoch + 1
        torch.save(model_BERT.state_dict(), "best_model_bert_standard.pth")
        print(f"Better model saved at epoch {best_metric_epoch} with val_accuracy: {best_metric:.4f}")

      wandb.log({"val_loss": avg_loss_val, "epoch": epoch})
      wandb.log({"val_accuracy": val_accuracy, "epoch": epoch})
      wandb.log({
        "val_loss": avg_loss_val,
        "val_accuracy": val_accuracy,
        "epoch": epoch
      })

      scheduler.step(avg_loss_val)
      print(f"Epoch {epoch+1}, Validation Loss: {avg_loss_val}, Validation Accuracy: {val_accuracy}")

  wandb.finish()

  """**EVALUATING BERT**"""

  wandb.init()
  wandb.login()

  model_BERT.load_state_dict(torch.load("best_model_bert_standard.pth"))
  model_BERT.to(device)
  model_BERT.eval()
  y_true_test = []
  y_pred_test = []
  total_test_loss = 0
  correct_test = 0
  total_test = 0

  with torch.no_grad():
    for batch in test_loader:
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels = batch["labels"].to(device)

      outputs = model_BERT(input_ids, attention_mask=attention_mask)
      loss = criterion(outputs.logits, labels)
      total_test_loss += loss.item()
      preds = torch.argmax(outputs.logits, dim=1)

      correct_test += (preds == labels).sum().item()
      total_test += labels.size(0)
      y_true_test.extend(labels.cpu().numpy())
      y_pred_test.extend(preds.cpu().numpy())


  avg_loss_test = total_test_loss / len(test_loader)
  test_accuracy = correct_test / total_test
  wandb.log({"test_loss": avg_loss_test, "test_accuracy": test_accuracy})

  print(f" Test Loss: {avg_loss_test:.4f}, Test Accuracy: {test_accuracy:.4f}")

  print("Classification Report on Test Set:")
  print(classification_report(y_true_test, y_pred_test))
   


"""**MT0 FUNCTION**"""

def MT0():
  """## **MT0 model** ##"""

  model_name = "bigscience/mt0-base"

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
          input_encoding = self.tokenizer( 
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

          return {
              "input_ids": input_encoding["input_ids"].squeeze(0),
              "attention_mask": input_encoding["attention_mask"].squeeze(0),
              "labels": label_encoding["input_ids"].squeeze(0)
          }

  label_map = {i: label for i, label in enumerate(label_encoder.classes_)} 
  dataset_train_mto = CustomTextDatasetMT0(X_train, Y_train, tokenizer, label_map) 
  dataset_test_mto = CustomTextDatasetMT0(X_test, Y_test, tokenizer, label_map) 
  dataset_val_mto = CustomTextDatasetMT0(X_val, Y_val, tokenizer, label_map) 

  test_loader_mto = DataLoader(dataset_test_mto, batch_size=16, shuffle=False, collate_fn=data_collator) 
  train_loader_mto = DataLoader(dataset_train_mto, batch_size=16, shuffle=True, collate_fn=data_collator) 
  val_loader_mto = DataLoader(dataset_val_mto, batch_size=16, shuffle=False, collate_fn=data_collator) 

  for batch in test_loader_mto:
      print(batch)
      break

  lr_param_mto = 5e-5 

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr_param_mto, weight_decay=0.01) 
  """**TRAINING MT0**"""

  """Aici in bucla de antrenare am detokenizat tokenii adica am transformat numere inapoi in stringuri ca sa compar cuvintele intre ele pentru ca:
  Prima oara am trecut etichetele printr-un label encoder adica le-am facut int-uri gen si dupa aceea le-am trecut printr-o clasa cu instantiere lazy care foloseste un tokenizer si gen aici am folosit un label_map care imi transforma inapoi din int in string si dupa ce sunt trecute prin clasa o sa am practic iarasi int-uri dar care sunt tokeni deci prin urmare am folosit un decodificator de la token ca sa transform numerele inapoi iarasi in string-uri ca sa le compar gen pentru ca MT0 foloseste eticheta ca cuvant propriu zis nu ca int.
  """

  model.to(device) 
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
  torch.cuda.empty_cache()
  wandb.login()

  lr = 0.001
  lr = lr_param_mto
  epochs = 25 # nr epoci
  epochs = 30
  epochs = 10
  epochs = 15

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



      model.train() 
      total_loss = 0 
      correct_train = 0
      total_train = 0

      for batch in train_loader_mto: 
          optimizer.zero_grad() 

          input_ids = batch["input_ids"].to(device) 
          attention_mask = batch["attention_mask"].to(device) 
          labels = batch["labels"].to(device) 

          outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) 
          loss = outputs.loss 

          loss.backward() 
          optimizer.step() 
          total_loss += loss.item() 
          preds = model.generate(input_ids, attention_mask=attention_mask) 

          decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
          decoded_labels = tokenizer.batch_decode(labels.detach().cpu().tolist(), skip_special_tokens=True)
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



      model.eval()
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) 
            loss = outputs.loss 
            total_val_loss += loss.item() 

            preds = model.generate(input_ids, attention_mask=attention_mask) 
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels.detach().cpu().tolist(), skip_special_tokens=True)
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
        torch.save(model.state_dict(), "best_model_mto.pth")
        print(f"Better model saved at epoch {best_metric_epoch} with val_accuracy: {best_metric:.4f}")
      wandb.log({
        "val_loss": avg_loss_val,
        "val_accuracy": val_accuracy,
        "epoch": epoch
      })

      scheduler.step(avg_loss_val)
      print(f"Epoch {epoch+1}, Validation Loss: {avg_loss_val}, Validation Accuracy: {val_accuracy}")

  wandb.finish()

  """**EVALUATING MT0**"""

  wandb.init()
  wandb.login()

  model.load_state_dict(torch.load("best_model_mto.pth"))
  model.to(device)
  model.eval()
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

      outputs =  model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss
      total_test_loss += loss
      preds = model.generate(input_ids, attention_mask=attention_mask) # , decoder_start_token_id=tokenizer.pad_token_id

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


"""##**BERT CALL**##"""
#BERT()

"""##**MT0 CALL**##"""
#MT0()



"""##**SEMI-SUPERVISED LEARNING**##"""

df_1mil = df = pd.read_json("dump.jsonl", lines=True)
print(type(df_1mil))

df_1mil.head(10)
label_map = {i: label for i, label in enumerate(label_encoder.classes_)} 
"""**TRAINING**"""

lr_param = 5e-6
tokenizer_BERT, data_collator_BERT, model_BERT, scheduler, optimizer, best_metric, best_metric_epoch = load_semi_supervised(lr_param)
criterion = torch.nn.CrossEntropyLoss() 
model_BERT.to(device) 
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

unlabeled_dataset = CustomTextDatasetSemi_Supervised(df_1mil, tokenizer_BERT)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False, collate_fn=data_collator_BERT)
unlabeled_loader_size = len(unlabeled_loader)
print(f"unlabeled_loader_size {unlabeled_loader_size}")



for batch in train_loader:
  print(batch)
  break


lr = lr_param
epochs = 10

best_metric = -1
best_metric_epoch = -1
train_accuracy_values = []
val_accuracy_values = []
confidence_threshold = 0.8 
predicted = {}
predicted = defaultdict(list)
class_names = [label_map[i] for i in range(num_labels)]

max_samples = 1000
max_samples_loader = 1000

num_iterations = 3 

for iteration in range(0, num_iterations):
  print(f"Iteration {iteration + 1}/{num_iterations}")
  print("---------------")
  wandb.login()
  run = wandb.init(
    project="semi-supervised bert",
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
    name=f"iteration - {iteration + 1}",
    reinit=True,
    force=True
  )

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

  lr_param = 5e-6
  tokenizer_BERT, data_collator_BERT, model_BERT, scheduler, optimizer, best_metric, best_metric_epoch = load_semi_supervised(lr_param)
  labels = [batch["labels"].cpu().numpy() for batch in train_loader]
  labels = np.concatenate(labels)

  class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
  class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

  criterion = torch.nn.CrossEntropyLoss()
  model_BERT.to(device)

  for epoch in range(epochs):
      print("---------------")
      print(f"Epoch {epoch + 1}/{epochs}")

      model_BERT.train() 
      total_loss = 0 
      correct_train = 0
      total_train = 0

      for batch in train_loader: 
          optimizer.zero_grad() 

          input_ids = batch["input_ids"].to(device) 
          attention_mask = batch["attention_mask"].to(device) 
          labels = batch["labels"].to(device) 

          outputs = model_BERT(input_ids, attention_mask=attention_mask) 
          loss = criterion(outputs.logits, labels) 

          loss.backward() 
          optimizer.step() 

          total_loss += loss.item() 
          preds = torch.argmax(outputs.logits, dim=1)
          correct_train += (preds == labels).sum().item()
          total_train += labels.size(0)
          wandb.log({"batch_loss": loss.item()})
          del outputs, loss
          torch.cuda.empty_cache()

      avg_loss_train = total_loss / len(train_loader)
      train_accuracy = correct_train / total_train
      train_accuracy_values.append(train_accuracy)

      wandb.log({
      "train_loss": avg_loss_train,
      "train_accuracy": train_accuracy,
      "epoch": int(epoch)
      })

      print(f"Epoch {epoch+1}, Training Loss: {avg_loss_train}, Training Accuracy: {train_accuracy}")


      model_BERT.eval()
      total_val_loss = 0
      correct_val = 0
      total_val = 0
      y_true = []
      y_pred = []

      with torch.no_grad():
        for batch in val_loader: 
            input_ids = batch["input_ids"].to(device) 
            attention_mask = batch["attention_mask"].to(device) 
            labels = batch["labels"].to(device) 

            outputs = model_BERT(input_ids, attention_mask=attention_mask) 
            loss = criterion(outputs.logits, labels) 
            total_val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())


      avg_loss_val = total_val_loss / len(val_loader)
      val_accuracy = correct_val / total_val

      if val_accuracy > best_metric:
        best_metric = val_accuracy
        best_metric_epoch = epoch + 1
        torch.save(model_BERT.state_dict(), "best_model_bert_standard.pth")
        print(f"Better model saved at epoch {best_metric_epoch} with val_accuracy: {best_metric:.4f}")

      wandb.log({
        "val_loss": avg_loss_val,
        "val_accuracy": val_accuracy,
        "epoch": int(epoch)
      })

      scheduler.step(avg_loss_val)
      print(f"Epoch {epoch+1}, Validation Loss: {avg_loss_val}, Validation Accuracy: {val_accuracy}")

     


  pseudo_labeled = []
  unlabeled = []
  unlabeled_text = []
  labeled_text = []

  model_BERT.eval()
  with torch.no_grad():

    for i, batch in enumerate(unlabeled_loader):
      print(f"batch {i + 1} / {unlabeled_loader_size}")
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)

      outputs = model_BERT(input_ids, attention_mask=attention_mask)
      probs = F.softmax(outputs.logits, dim=1)
      confs, preds = torch.max(probs, dim=1)
      print(f"[DEBUG] Max confidence în batch: {confs.max().item():.4f}")
      print(f"[DEBUG] Nr. exemple peste threshold: {(confs > confidence_threshold).sum().item()}")
      for i in range(0, len(preds)):
        if confs[i].item() >= confidence_threshold:
          pseudo_labeled.append({
              "input_ids": input_ids[i].cpu(),
              "attention_mask": attention_mask[i].cpu(),
              "label": preds[i].cpu(),
              "confidence": confs[i].item()
          })
        else:
          unlabeled.append({
              "input_ids": input_ids[i].cpu(),
              "attention_mask": attention_mask[i].cpu()
          })
              
      

  print(f" Am adăugat {len(pseudo_labeled)} pseudo-etichetări.")
  if len(pseudo_labeled) == 0:
    print("Nicio etichetă suficient de sigură. Ne oprim.")
    break
  
  """**EVALUATING BERT**"""
  model_BERT.load_state_dict(torch.load("best_model_bert_standard.pth"))
  model_BERT.to(device)
  model_BERT.eval()
  y_true_test = []
  y_pred_test = []
  total_test_loss = 0
  correct_test = 0
  total_test = 0

  with torch.no_grad():
    for batch in test_loader:
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels = batch["labels"].to(device)

      outputs = model_BERT(input_ids, attention_mask=attention_mask)
      loss = criterion(outputs.logits, labels)
      total_test_loss += loss.item()
      preds = torch.argmax(outputs.logits, dim=1)

      correct_test += (preds == labels).sum().item()
      total_test += labels.size(0)
      y_true_test.extend(labels.cpu().numpy())
      y_pred_test.extend(preds.cpu().numpy())


  avg_loss_test = total_test_loss / len(test_loader)
  test_accuracy = correct_test / total_test
  
  wandb.log({"test_loss": avg_loss_test,
            "test_accuracy": test_accuracy,
            "epoch": int(epoch)})
  
  f1_value = f1_score(y_true=y_true_test, y_pred=y_pred_test, average="macro")
  precision_value = precision_score(y_true=y_true, y_pred=y_pred, zero_division=0, average="macro")
  recall_value = recall_score(y_true=y_true, y_pred=y_pred, zero_division=0, average="macro")
  wandb.log({"f1-score": f1_value, 
             "precision-score": precision_value, 
             "recall-score": recall_value, 
             "iteration": int(iteration)
            })
  
  cm_standard = confusion_matrix(y_true=y_true_test, y_pred=y_pred_test)
  plt.figure(figsize=(20, 20))
  sns.heatmap(cm_standard, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8})
  
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix STANDARD BERT')

  plt.savefig('confusion_matrix_bert_standard.png', dpi=300, bbox_inches='tight')

  print(f" Test Loss: {avg_loss_test:.4f}, Test Accuracy: {test_accuracy:.4f}")

  print("Classification Report on Test Set:")
  print(classification_report(y_true_test, y_pred_test)) 

  class_counts = defaultdict(list)

  for ex in pseudo_labeled:
    label = ex["label"].item()
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

  pseudo_labeled = final_pseudo

  train_data = list(train_loader.dataset)
  train_data.extend(pseudo_labeled)

  train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=data_collator_BERT) 

  unlabeled_loader = DataLoader(unlabeled, batch_size=32, shuffle=False, collate_fn=data_collator_BERT)
  unlabeled_loader_size = len(unlabeled_loader)
  
  if confidence_threshold >= 0.65:
    confidence_threshold -= 0.05


  wandb.finish()

max_displayed = 10

with open("predicted_top10_texts_semi_supervised_standard.csv", mode = 'w', newline="", encoding="utf-8") as file:
  writer=csv.writer(file)
  writer.writerow(["label"] + [f"text_{i+1}" for i in range(max_displayed)] + [f"conf_{i+1}" for i in range(max_displayed)])

  for label, examples in predicted.items(): # class_counts_deepcopy
    sorted_examples = sorted(examples, key=lambda x: x["confidence"], reverse=True)
    top_10 = sorted_examples[:max_displayed]

    texts = [tokenizer_BERT.decode(ex["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True) for ex in top_10]
    confidences = [f"{ex['confidence']:.4f}" for ex in top_10]

    texts += [""] * (max_displayed - len(texts))
    confidences += [""] * (max_displayed - len(confidences))

    writer.writerow([label_map[label]] + texts + confidences)


print("\n Am generat csv-ul !!! \n")