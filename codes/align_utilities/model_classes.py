import random
import glob
import os
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from numpy.linalg import norm
import utilities as utl
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, Sigmoid
from torch.optim import *
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, RobertaTokenizerFast, RobertaModel

# define the network
class BertClassifierPretrained(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        # Freeze/ unfreeze all the parameters in the BERT model
        for param in self.bert.parameters():
            param.requires_grad = False
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        pooled_output = outputs.pooler_output # outputs[1]
        return pooled_output # use this for pre-trained model result

