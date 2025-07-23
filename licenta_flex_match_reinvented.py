import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import nltk
import nlpaug.augmenter.word as naw
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
import time, psutil
from collections import defaultdict
from itertools import cycle
import copy
from transformers import BertTokenizer
from transformers import DataCollatorWithPadding

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

def BERT():
  for batch in train_loader:
      print(batch)
      break

  print(f"train_loader labels : {len(train_loader.dataset.labels)}")
  print(f"val_loader labels : {len(val_loader.dataset.labels)}")

  """**TRAINING BERT**"""

  model_BERT.to(device)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

  
  lr = lr_param
  epochs = 3 
  
  best_metric = -1
  best_metric_epoch = -1
  train_accuracy_values = []
  val_accuracy_values = []

  print(f"lr: {lr}")

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

          del outputs, loss
          gc.collect()
          torch.cuda.empty_cache()

      avg_loss_train = total_loss / len(train_loader)
      train_accuracy = correct_train / total_train
      train_accuracy_values.append(train_accuracy)

      print(f"Epoch {epoch+1}, Training Loss: {avg_loss_train}, Training Accuracy: {train_accuracy}")


      # Validarea

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
        torch.save(model_BERT.state_dict(), "best_model_bert_flex.pth")
        print(f"Better model saved at epoch {best_metric_epoch} with val_accuracy: {best_metric:.4f}")

      scheduler.step(avg_loss_val)
      print(f"Epoch {epoch+1}, Validation Loss: {avg_loss_val}, Validation Accuracy: {val_accuracy}")


  """**EVALUATING BERT**"""

  model_BERT.load_state_dict(torch.load("best_model_bert_flex.pth"))
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

  print(f" Test Loss: {avg_loss_test:.4f}, Test Accuracy: {test_accuracy:.4f}")

  print("Classification Report on Test Set:")
  print(classification_report(y_true_test, y_pred_test, zero_division=0))

BERT()

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

unlabeled_dataset = CustomTextDatasetSemi_Supervised(df_1mil, tokenizer_BERT)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False, collate_fn=data_collator_BERT, num_workers=num_workers, persistent_workers=True)
unlabeled_loader_size = len(unlabeled_loader)
print(f"unlabeled_loader_size {unlabeled_loader_size}")

def ensure_tensor(example):
    for key in ["input_ids", "attention_mask", "labels"]:
        if not isinstance(example[key], torch.Tensor):
            example[key] = torch.tensor(example[key], dtype=torch.long)

for batch in train_loader:
  print(batch)
  print(f"type batch : {type(batch)}")
  break



lr = lr_param


best_metric = -1
best_metric_epoch = -1
train_accuracy_values = []
val_accuracy_values = []
predicted = defaultdict(list)

lambda_u = 1.0
max_samples = 1000
max_samples_loader = 1000

num_iterations = 8 
epochs = 1 

augmenter_weak = naw.RandomWordAug(action="swap", aug_p=0.05)  # 5% din cuvinte swap
augmenter_strong = naw.RandomWordAug(action="swap", aug_p=0.1)  # 10% din cuvinte swap
thresholds = torch.full((num_labels,), 0.8).to(device) # 0.8 # 0.7
optimizer = torch.optim.AdamW(model_BERT.parameters(), lr=lr_param, weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
criterion = torch.nn.CrossEntropyLoss()
class_names = [label_map[i] for i in range(num_labels)]

wandb.login()

run = wandb.init(
project="semi-supervised bert",
config={
    "learning_rate": lr,
    "iterations": num_iterations,
  },
  name=f"full semi-supervised FlexMatch",
  reinit=True 
)

wandb.define_metric("iteration", step_metric=None)
wandb.define_metric("train_loss", step_metric="iteration")
wandb.define_metric("train_accuracy", step_metric="iteration")
wandb.define_metric("val_loss", step_metric="iteration")
wandb.define_metric("val_accuracy", step_metric="iteration")
wandb.define_metric("batch_loss", step_metric="iteration")
wandb.define_metric("unsupervised_loss", step_metric="iteration")
wandb.define_metric("supervised loss", step_metric="iteration")
wandb.define_metric("test_loss", step_metric="iteration")
wandb.define_metric("test_accuracy", step_metric="iteration")
wandb.define_metric("f1-score", step_metric="iteration")
wandb.define_metric("precision-score", step_metric="iteration")
wandb.define_metric("recall-score", step_metric="iteration")

offset = random.random() / 5
print(f"lr: {lr}")

for iteration in range(0, num_iterations):
  
  print(f"Iteration {iteration + 1}/{num_iterations}")
  print("---------------")
  lr_param = 5e-6
  full_unlabeled_dataset = list(unlabeled_loader.dataset)
  
  classwise_acc = torch.zeros(num_labels).to(device)
  
  pseudo_labeled = []
  unlabeled = []

  labels = [batch["labels"].cpu().numpy() for batch in train_loader]
  labels = np.concatenate(labels)

  for epoch in range(epochs):
      print("---------------")
      print(f"Epoch {epoch + 1}/{epochs}")

      model_BERT.train()
      total_loss = 0 
      correct_train = 0
      total_train = 0
      
      batch = None 
      
      train_iter = cycle(train_loader)
      steps_per_batch = 1000
      sigma = torch.zeros(num_labels).to(device) * 1e-3
      batch = next(train_iter)
        

      for i, unlabeled_batch in enumerate(unlabeled_loader):
          print(f"At iteration {iteration + 1} from epoch {epoch + 1} : Batch {i + 1} \ {len(unlabeled_loader)}")

          if i >= steps_per_batch:
             break

          optimizer.zero_grad()
          input_ids = batch["input_ids"].to(device)
          attention_mask = batch["attention_mask"].to(device)
          labels = batch["labels"].to(device)

          outputs = model_BERT(input_ids, attention_mask=attention_mask)
          print("trecut de forward pass \n")
          
          #FlexMatch
          with torch.no_grad():
            decoded_texts = [
                tokenizer_BERT.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for ids in unlabeled_batch["input_ids"]
            ]
            print("a terminat de decodat \n")
            assert len(decoded_texts) == len(unlabeled_batch["input_ids"]), "[DEBUG] Decoded texts and input_ids mismatch!"

            start = time.time()
            
            augmented_weak_texts = []
            for txt in decoded_texts:
              augmented_texts = augmenter_weak.augment(txt)
              if augmented_texts:
                augmented_weak_texts.append(augmented_texts[0])
              else:
                 augmented_weak_texts.append(txt)
        
            elapsed = time.time() - start

            ram_used = psutil.virtual_memory().used / 1e9
            print(f"[DEBUG] Timp augmentare: {elapsed:.2f}s | RAM: {ram_used:.2f} GB")
            inputs_weak = tokenizer_BERT(augmented_weak_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            print(f"[DEBUG] Weak tokenized shape: {inputs_weak['input_ids'].shape}")
          with torch.no_grad():
            logits_weak = model_BERT(input_ids=inputs_weak["input_ids"], attention_mask=inputs_weak["attention_mask"]).logits

          probs_weak = F.softmax(logits_weak, dim=-1)
          max_probs, pseudo = torch.max(probs_weak, dim=-1)
          mask = max_probs > thresholds[pseudo]
          print("am luat mastile \n")

          
          for i in range(len(pseudo)):
            cls = pseudo[i].item()
            correct = 0.0
            if mask[i]:
                sigma[cls] += 1
                pseudo_labeled.append({
                    "input_ids": unlabeled_batch["input_ids"][i].cpu(),
                    "attention_mask": unlabeled_batch["attention_mask"][i].cpu(),
                    "labels": pseudo[i].cpu(),
                    "confidence": max_probs[i].item()
                })
                correct = 1.0
            else: # nesigur
              unlabeled.append({
                "input_ids": unlabeled_batch["input_ids"][i].cpu(),
                "attention_mask": unlabeled_batch["attention_mask"][i].cpu()
              })
              correct = 0.0

            classwise_acc[cls] = 0.9 * classwise_acc[cls] + 0.1 * correct
            
          print(f"pana acum in pseudo_labeled avem : {len(pseudo_labeled)}")
          
          selected_strong_ones = [decoded_texts[i] for i in range(len(decoded_texts)) if mask[i]]
          
          logits_strong = torch.empty((0, num_labels), device=device)

          if selected_strong_ones:
            augmented_strong_texts = []
            for txt in selected_strong_ones:
              augmented_strong_txts = augmenter_strong.augment(txt)
              if augmented_strong_txts:
                augmented_strong_texts.append(augmented_strong_txts[0])
              else:
                augmented_strong_texts.append(txt)
            inputs_strong = tokenizer_BERT(augmented_strong_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)

            logits_strong = model_BERT(input_ids=inputs_strong["input_ids"], attention_mask=inputs_strong["attention_mask"]).logits

            pseudo_strong = pseudo[mask] 
            print("calculam loss-uri acum !!! ")
            loss_u = F.cross_entropy(logits_strong, pseudo_strong) if pseudo_strong.numel() > 0 else torch.tensor(0.0, device=device)
          else:
            loss_u = torch.tensor(0.0, device=device)
          
          logits_sup = model_BERT(input_ids, attention_mask=attention_mask).logits
          loss_sup = criterion(logits_sup, labels)

          loss = loss_sup + lambda_u * loss_u

          wandb.log({
             "unsupervised_loss" : loss_u,
             "supervised loss" : loss_sup,
             "iteration" : int(iteration)
          })

          if loss.item() != 0: 
            batch = next(train_iter)
            loss.backward() 
            optimizer.step()
        
          optimizer.zero_grad()

          total_loss += loss.item()
          preds = torch.argmax(outputs.logits, dim=1)
          correct_train += (preds == labels).sum().item()
          total_train += labels.size(0)
          
          wandb.log({"batch_loss": loss.item(), "iteration" : int(iteration)})
          
          del outputs, logits_weak, probs_weak, pseudo, mask, loss_sup, loss_u, loss, input_ids, attention_mask, labels # , batch
          del inputs_weak, augmented_weak_texts

          if 'inputs_strong' in locals():
              del inputs_strong
          if 'augmented_strong_texts' in locals():
              del augmented_strong_texts
          gc.collect()
          torch.cuda.empty_cache()

          avg_loss_train = total_loss / len(train_loader)
          train_accuracy = correct_train / total_train
          train_accuracy_values.append(train_accuracy)



          wandb.log({"threshold_evolution" : thresholds, "iteration" : int(iteration)})

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
        torch.save(model_BERT.state_dict(), "best_model_bert_flex.pth")
        print(f"Better model saved at epoch {best_metric_epoch} with val_accuracy: {best_metric:.4f}")

      scheduler.step(avg_loss_val)

  wandb.log({
    "thresholds_hist": wandb.Histogram(thresholds.cpu().numpy()),
    "iteration": int(iteration)
  })
  
  N = len(unlabeled_dataset)
  beta = sigma / torch.clamp(torch.maximum(sigma.max(), N - sigma.sum()), min=1e-6)
  thresholds = (beta / (2 - beta)) * 0.9 # 0.95

  wandb.log({
    "train_loss": avg_loss_train,
    "train_accuracy": train_accuracy,
    "val_loss": avg_loss_val,
    "val_accuracy": val_accuracy,
    "iteration": int(iteration)
  })

  
  """**EVALUATING BERT**"""
  model_BERT.load_state_dict(torch.load("best_model_bert_flex.pth"))
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
  wandb.log({"test_loss": avg_loss_test, "test_accuracy": test_accuracy, "iteration" : int(iteration)})

  print(f" Test Loss: {avg_loss_test:.4f}, Test Accuracy: {test_accuracy:.4f}")

  print("Classification Report on Test Set:")
  print(classification_report(y_true_test, y_pred_test))
  f1_value = f1_score(y_true=y_true_test, y_pred=y_pred_test, average="macro")
  precision_value = precision_score(y_true=y_true_test, y_pred=y_pred_test, zero_division=0, average="macro")
  recall_value = recall_score(y_true=y_true_test, y_pred=y_pred_test, zero_division=0, average="macro")
  wandb.log({"f1-score" : f1_value,
             "precision-score" : precision_value,
             "recall-score" : recall_value,
             "iteration" : int(iteration)})
  
  cm_flex_match = confusion_matrix(y_true=y_true_test, y_pred=y_pred_test)
  plt.figure(figsize=(20, 20))
  sns.heatmap(cm_flex_match, annot=True, fmt='d', cmap='Blues',
      xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8})
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix FLEX MATCH')

  plt.savefig('confusion_matrix_flex_match.png', dpi=300, bbox_inches='tight')
   

  class_counts = defaultdict(list)

  for ex in pseudo_labeled:
    label = ex["labels"].item()
    class_counts[label].append(ex)

  final_pseudo = []
  
  for label, examples in class_counts.items():
    sorted_examples = sorted(examples, key=lambda x: x["confidence"], reverse=True)
    print(f"pentru clasa {label_map[label]} am scos {len(sorted_examples)}")
    final_pseudo.extend(sorted_examples[:((3 * len(sorted_examples)) // 4)]) # 1000
    predicted[label].extend(copy.deepcopy(examples))
  class_counts_deepcopy = copy.deepcopy(class_counts)
  
  for example in final_pseudo:
    if "confidence" in example:
      del example["confidence"]
    
    ensure_tensor(example)

  pseudo_labeled = final_pseudo

  seen = set()

  for ex in pseudo_labeled:
      key = (
          tuple(ex["input_ids"].tolist()),
          tuple(ex["attention_mask"].tolist())
      )
      seen.add(key)

  unlabeled = [
      ex for ex in full_unlabeled_dataset
      if (
          tuple(ex["input_ids"].tolist()),
          tuple(ex["attention_mask"].tolist())
      ) not in seen
  ]

  train_data = list(train_loader.dataset)
  train_data.extend(pseudo_labeled)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=data_collator_BERT, num_workers=num_workers, persistent_workers=False) # TRUE sau FALSE

  for i, batch in enumerate(train_loader):
    assert "labels" in batch, f"Missing 'labels' in batch {i}"
    assert isinstance(batch["labels"], torch.Tensor), f"'labels' not a tensor in batch {i}"
    break

  unlabeled_loader = DataLoader(unlabeled, batch_size=32, shuffle=False, collate_fn=data_collator_BERT, num_workers=num_workers, persistent_workers=True)
  unlabeled_loader_size = len(unlabeled_loader)



wandb.finish()

max_displayed = 10

with open("predicted_top10_texts_semi_supervised_Flex_Match.csv", mode = 'w', newline="", encoding="utf-8") as file:
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
