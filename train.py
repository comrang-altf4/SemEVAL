from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from datasets import Dataset
from sentence_transformers import SentenceTransformer,  losses, InputExample, util
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np

# read csv file
df_n1 = pd.read_csv('./narratives_topic1.csv')
df_n2 = pd.read_csv('./narratives_topic2.csv')
df_s1 = pd.read_csv('./sub_narratives_topic1.csv')
df_s2 = pd.read_csv('./sub_narratives_topic2.csv')
df_s2['Narrative ID'] = df_s2['Narrative ID'] + len(df_n1)
df_s = pd.concat([df_s1, df_s2], axis=0)
df_s = df_s.reset_index(drop=True)
df_s['Narrative ID'] = df_s['Narrative ID']-1
df_n = pd.concat([df_n1, df_n2], axis=0)
df_n = df_n.reset_index(drop=True)
# lowercase all the text
df_n['Narrative'] = df_n['Narrative'].str.lower()
df_s['Sub-Narrative'] = df_s['Sub-Narrative'].str.lower()

reference_df = pd.read_csv('./EN/subtask-2-annotations.txt', sep='\t', header=None)
reference_df[1] = reference_df[1].str.lower()
reference_df[2] = reference_df[2].str.lower()

reference_df['Narrative ID'] = None
for i in range(len(reference_df)):
    narrative_name = reference_df.iloc[i, 1]
    narrative_name = narrative_name.split(';')
    ids =[]
    for name in narrative_name:
        if 'other' in name:
            ids.append(None)
            continue
        for j in range(len(df_n)):
            if df_n.iloc[j, 0] in name:
                ids.append(j)
    reference_df['Narrative ID'].loc[i] = ids

reference_df['Sub_Narrative ID'] = None
for i in range(len(reference_df)):
    narrative_name = reference_df.iloc[i, 2]
    narrative_name = narrative_name.split(';')
    ids =[]
    for name in narrative_name:
        if 'other' in name:
            ids.append(None)
            continue
        for j in range(len(df_s)):
            if df_s.iloc[j, 0] in name:
                ids.append(j)
    reference_df['Sub_Narrative ID'].loc[i] = ids
reference_df = reference_df.drop(columns=[1, 2])
def load_article(file_path):
    with open(file_path, 'r',encoding="utf8") as file:
        data = file.read().replace('\n', '')
    return data

article_dir = './EN/raw-documents/'
article_files = os.listdir(article_dir)
article_files = sorted(article_files)

article_data = {}
for file in article_files:
    article_data[file] = load_article(article_dir + file)
df_n = df_n.fillna('None')
df_s = df_s.fillna('None')

all_pairs = []
all_narrative_id = df_s['Narrative ID'].unique()
all_sub_narrative_id = df_s.index
for idx,row in tqdm(reference_df.iterrows(), total=len(reference_df)):
    article1 = article_data[row[0]]
    for i in all_narrative_id:
        article2 = f"Narrative type: {df_n.iloc[i, 0]}. Narrative description: {df_n.iloc[i, 1]}. Instruction for annotators: {df_n.iloc[i, 2]}."
        label = 1 if i in row['Narrative ID'] else 0 
        all_pairs.append((article1, article2, label))
    for i in all_sub_narrative_id:
        article2 = f"Sub-Narrative type: {df_s.iloc[i, 0]}. Sub-Narrative description: {df_s.iloc[i, 1]}. Instruction for annotators: {df_s.iloc[i, 2]}."
        label = 1 if i in row['Sub_Narrative ID'] else 0 
        all_pairs.append((article1, article2, label))

# 1. Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')
# 7. Train-test split
dataset_articles = [pair[0] for pair in all_pairs]
dataset_narratives = [pair[1] for pair in all_pairs]
dataset_labels = [pair[2] for pair in all_pairs]
dataset = Dataset.from_dict({
    "articles": dataset_articles,
    "narratives": dataset_narratives,
    "label": dataset_labels
})

dataset_split = dataset.train_test_split(test_size=0.2,seed = 42)
#check torch cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(device)
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/mpnet-base-all-nli-triplet",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    # run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
)
train_loss = losses.ContrastiveLoss(model)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=dataset_split['train'],
    loss=train_loss,
)
trainer.train()
model.save_pretrained("models/mpnet-base-all-nli-triplet/final")