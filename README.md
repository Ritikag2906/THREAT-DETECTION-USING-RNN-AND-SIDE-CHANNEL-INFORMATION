# THREAT-DETECTION-USING-RNN-AND-SIDE-CHANNEL-INFORMATION
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt 
src = "/kaggle/input/dataset-major-project/data_blend"

paths = ["{}/{}".format(src, x) for x in os.listdir(src) if x[-3:] == "csv"]
labels = ["benign" if x.split("/")[-1].split("_")[0] == "benign" else "malware" for x in paths]

df = pd.DataFrame({"Path": paths, "Label": labels})
df.head()
# maximum number of rows for given file
maxlen = 0

# merging rows for all files for getting all values for categorical columns
fulldf = pd.DataFrame()
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    dfx = pd.read_csv(row["Path"])
    maxlen = max(dfx.shape[0], maxlen)
    fulldf = pd.concat([fulldf, dfx], axis=0)
    # columns to be removed due to irrelevance, we remove activity as it gives away the label
remcols = ["Date", "Time", "OS", "activity", "Label"]

# columns with a single value across all files will also be removed
for col in fulldf.columns:
    if fulldf[col].nunique() == 1 and col not in remcols:
        remcols.append(col)
# removing columns before generating encoder for necessary categorical columns
fulldf = fulldf.drop(remcols, axis=1)
# getting all the encoders ready for transforming categorical columns
encoders = {}
for col in tqdm(fulldf.select_dtypes(include=['object']).columns):
    encoder = LabelEncoder()
    encoder.fit(fulldf[col])
    encoders[col] = encoder
    Train, Valid = train_test_split(df, test_size=0.4, stratify=df["Label"], random_state=29)
    for x, y  in zip([Train, Valid], ["Train", "Valid"]):
    label_count = x["Label"].value_counts()
    print(y, "| M:", label_count["malware"], "B:", label_count["benign"])
    train_embdf = {"Label": []}
embedding_size = 100
for i in range(embedding_size):
    train_embdf["emb_{}".format(i)] = []


rnn = nn.LSTM(97, embedding_size, 1, batch_first=True) #long short term memory
rnn.to("cuda:0")
for idx, row in tqdm(Train.iterrows(), total=Train.shape[0]):
    csv = pd.read_csv(row["Path"]).drop(remcols, axis=1)
    r, c = csv.shape
    for key in encoders.keys():
        csv[key] = encoders[key].transform(csv[key])
    features = torch.tensor(csv.values, dtype=torch.float32).unsqueeze(0).cuda()
    h0 = torch.zeros(1, features.size(0), embedding_size).float().cuda()
    c0 = torch.zeros(1, features.size(0), embedding_size).float().cuda()
    output = rnn(features, (h0, c0))[0][:, -1, :]
    output = output.detach().cpu().numpy()

    for i, x in enumerate(output[0]):
        train_embdf["emb_{}".format(i)].append(x)
    label = "benign" if row["Path"].split("/")[-1].split("_")[0] == "benign" else "malware"
    train_embdf["Label"].append(label)
train_embdf = pd.DataFrame(train_embdf)
train_embdf.to_csv("train_embedding.csv", index=False)
train_embdf.head()
