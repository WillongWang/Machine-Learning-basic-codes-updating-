# Import necessary libraries
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load and shuffle dataset
df = pd.read_csv("IMDB Dataset.csv")
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split data into training and testing sets
train_size = int(0.8 * len(df)) #len(DataFrame)指其的行数（records数，除了第一行标签行，也许是因为DataFrame类似二维数组（可能是list或ndarray），实验一下吧）
test_size = len(df) - train_size
train_dataset, test_dataset = random_split(df, [train_size, test_size]) #random_split也许把作为dataset的DataFrame的每一行record（除第一行标签行）作为DataFrame集合的元素
# get length of all the records in the "review" colomn
seq_len = [len(i.split()) for i in df['review']]
pd.Series(seq_len).hist(bins = 30)
plt.show()
maxlen=512 #max for bert

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #返回的就是BertTokenizer吧

class IMDbDataset(Dataset): #？？？说实话不懂为什么可以这么定义IMDbDataset，最后在Trainer中也没有显式提取出train_dataset的embedding,mask,labels进行计算
                                  #估计只有看会Trainer和BertForSequenceClassification源码才懂，但不可能看的会啊
    def __init__(self, df, tokenizer, maxlen):
        self.review = list(df['review'])
        self.labels = list(df['label'])
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __getitem__(self, idx):
        # Tokenize each sample individually to apply `maxlen`
        encoding = self.tokenizer( #看__call__
            self.review[idx],
            padding='max_length',
            truncation=True,
            max_length=self.maxlen, #指每个DataFrame的record（每一行除第一个标签行）的text（非label）的最大token长度
            return_tensors="pt"
        )
        item = {key: val.squeeze().to(device) for key, val in encoding.items()} #encoding是个字典，类型是BatchEncoding，据huggingface transformers官网，
                                                #首先找到class transformers.PreTrainedTokenizerBase，__call__函数栏有一个A BatchEncoding with the following fields:
                                                #下面的粗体字应该就是encoding的key，对应的值即对应粗体字后-的右边，若return_tensors="pt"，-右边变为tensor
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)

full_dataset = IMDbDataset(df, tokenizer, maxlen)

# Split data into training and testing sets
train_size = int(0.8 * len(df))
test_size = len(df) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Evaluation metrics
def compute_metrics(pred): #compute_metrics在Trainer参数里，compute_metrics参数为EvalPrediction
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1) #gpt的回答：pred.predictions is a 2D tensor of logits with dimensions [batch_size, num_labels]. 
    #Each row corresponds to the logits for a particular sample, with each column representing a score for a specific label
    #The argmax(-1) operation selects the index of the highest logit along the last dimension, num_labels, for each sample in the batch. This gives a 1D array of predicted labels for all samples in the batch. 
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Define experiments
experiments = [
    {"freeze_layers": 0, "learning_rate": 3e-5, "batch_size": 32, "epochs": 2, "dropout": 0.1, "use_cls_token": True},
    {"freeze_layers": 0, "learning_rate": 3e-5, "batch_size": 16, "epochs": 2, "dropout": 0.1, "use_cls_token": True},
    {"freeze_layers": 0, "learning_rate": 3e-5, "batch_size": 32, "epochs": 3, "dropout": 0.1, "use_cls_token": True},
    {"freeze_layers": 0, "learning_rate": 5e-5, "batch_size": 32, "epochs": 2, "dropout": 0.1, "use_cls_token": True},
    {"freeze_layers": 0, "learning_rate": 5e-5, "batch_size": 32, "epochs": 3, "dropout": 0.1,  "use_cls_token": True},
    {"freeze_layers": 0, "learning_rate": 3e-5, "batch_size": 32, "epochs": 2, "dropout": 0.2,  "use_cls_token": True},
    {"freeze_layers": 4, "learning_rate": 3e-5, "batch_size": 32, "epochs": 2, "dropout": 0.1,  "use_cls_token": True},
    {"freeze_layers": 8, "learning_rate": 3e-5, "batch_size": 32, "epochs": 2, "dropout": 0.1,  "use_cls_token": True},
    {"freeze_layers": 12, "learning_rate": 3e-5, "batch_size": 32, "epochs": 2, "dropout": 0.1,  "use_cls_token": True},
    {"freeze_layers": 12, "learning_rate": 3e-5, "batch_size": 32, "epochs": 2, "dropout": 0.2,  "use_cls_token": False},
]

# Run experiments
results = []
for idx, exp in enumerate(experiments):
    print(f"Running experiment {idx + 1}")
    
    # Load pre-trained BERT model
    config = BertConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=exp["dropout"], num_labels=2)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
    #model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2) #官网中BertForSequenceClassification下面的示例有，from_pretrained无config参数，虽然没找到from_pretrained有num_labels参数
    model = model.to(device)

    # Freeze layers as specified
    for layer in model.bert.encoder.layer[:exp["freeze_layers"]]: #看一下BertForSequenceClassification、BertModel、BertEncoder源码你就知道有bert.encoder.layer,config.num_hidden_layers = 12
        for param in layer.parameters():
            param.requires_grad = False

    # Define training arguments
    training_args = TrainingArguments(
        dataloader_pin_memory=False,
        output_dir=f"./results/experiment_{idx + 1}",
        num_train_epochs=exp["epochs"],
        per_device_train_batch_size=exp["batch_size"],
        per_device_eval_batch_size=exp["batch_size"],
        learning_rate=exp["learning_rate"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=50,  # More frequent logging for better tracking
        lr_scheduler_type="linear"  # Optional learning rate schedule
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics #官方示例中不用写compute_metrics的参数
    )

    # Train and evaluate
    trainer.train()
    eval_result = trainer.evaluate() #返回compute_metrics的返回值
    results.append({"experiment": idx + 1, "config": exp, "eval_result": eval_result})
    print(f"Experiment {idx + 1} complete with result: {eval_result}")

# Print all experiment results
for result in results:
    print(result)
