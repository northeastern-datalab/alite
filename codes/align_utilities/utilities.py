import random
import pickle
import bz2
import os
import sys
import csv
import torch
import numpy as np
import pandas as pd
import _pickle as cPickle
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

# Useful functions copied from SANTOS
# --------------------------------------------------------------------------------
# This function saves dictionaries as pickle files in the storage.
def saveDictionaryAsPickleFile(dictionary, dictionaryPath):
    if dictionaryPath.rsplit(".")[-1] == "pickle":
        filePointer=open(dictionaryPath, 'wb')
        pickle.dump(dictionary,filePointer, protocol=pickle.HIGHEST_PROTOCOL)
        filePointer.close()
    else: #pbz2 format
        with bz2.BZ2File(dictionaryPath, "w") as f: 
            cPickle.dump(dictionary, f)

            
# load the pickle file as a dictionary
def loadDictionaryFromPickleFile(dictionaryPath):
    print("Loading dictionary at:", dictionaryPath)
    if dictionaryPath.rsplit(".")[-1] == "pickle":
        filePointer=open(dictionaryPath, 'rb')
        dictionary = pickle.load(filePointer)
        filePointer.close()
    else: #pbz2 format
        dictionary = bz2.BZ2File(dictionaryPath, "rb")
        dictionary = cPickle.load(dictionary)
    print("The total number of keys in the dictionary are:", len(dictionary))
    return dictionary


# load csv file as a dictionary. Further preprocessing may be required after loading
def loadDictionaryFromCsvFile(filePath):
    if(os.path.isfile(filePath)):
        with open(filePath) as csv_file:
            reader = csv.reader(csv_file)
            dictionary = dict(reader)
        return dictionary
    else:
        print("Sorry! the file is not found. Please try again later. Location checked:", filePath)
        sys.exit()
        return 0
    
# --------------------------------------------------------------------------------
# New functions specific to this project
# --------------------------------------------------------------------------------
# A function to compute cosine similarity between two numpy arrays
def CosineSimilarity(array1, array2):
    return np.dot(array1,array2)/(norm(array1)*norm(array2))

# A function that takes a table as pandas dataframe and returns a list of its serialized rows. Each row is serialized as a separate sentence.
# Serialization format: COL <col1 name> VAL <col1 value> COL <col2 name> VAL <col2 value> ..... COL <colN name> VAL <colN value>
def SerializeTable(table_df):
    rows = table_df.to_dict(orient='records')
    serialized_rows = []
    for item in rows:
        current_serialization = SerializeRow(item)
        serialized_rows.append(current_serialization)
    return serialized_rows

# input_sentence = "COL column1 name VAL column1 value COL column2 name VAL column2 value COL column3 name VAL column3 value"
def UseSEPToken(sentence):
    # Split the input sentence into pairs of column name and value
    pairs = sentence.split('COL')[1:]
    # Create the transformed sentence
    transformed_sentence = "[CLS] " + " [SEP] ".join(" ".join(pair.strip().replace("VAL", "").split(" ")) for pair in pairs) + " [SEP]"
    transformed_sentence = transformed_sentence.strip()
    return transformed_sentence

def SerializeRow(row):
    current_serialization = str()
    for col_name in row:
        cell_value = str(row[col_name]).replace("\n","").replace("\t", " ")
        col_name = str(col_name).replace("\n", "").replace("\t"," ")
        current_serialization += "COL " + col_name + " VAL " + cell_value + " "
    current_serialization = current_serialization.strip() #remove trailing and leading spaces
    current_serialization = current_serialization.replace("\n", "")
    current_serialization = UseSEPToken(current_serialization) #remove this line to use old serialization
    return current_serialization

# A function that takes a list of serialized rows as input and returns embeddings for the table.
# It computes average embedding of a sample of rows, adds new rows iteratively to the sample and recompute embeddings.
# The table embedding is confirmed when the stopping criteria is reached i.e., the newly added samples are not impacting the embeddings by already selected samples.
def EmbedTable(serialized_rows, model, embedding_type, tokenizer, sample_size = 20, sim_threshold = 0.05):
    total_rows = len(serialized_rows)
    used_rows = 0
    #serialized_rows = set(serialized_rows) #using set of rows so that we can quickly sample without replacement
    sample1_list = random.sample(serialized_rows, min(sample_size, len(serialized_rows)))
    if embedding_type == "sentence_bert":
        sample1_embeddings = model.encode(sample1_list)
    else: #add more for other kinds
        sample1_embeddings = encode_finetuned(sample1_list, model, tokenizer)
    sample1_average_embeddings = np.mean(sample1_embeddings, axis=0)
    serialized_rows = list(set(serialized_rows) - set(sample1_list))
    while(len(serialized_rows) > 0):
        sample2_list = random.sample(serialized_rows, min(sample_size, len(serialized_rows)))
        if embedding_type == "sentence_bert":
            sample2_embeddings = model.encode(sample2_list)
        else:
            sample2_embeddings = encode_finetuned(sample2_list, model, tokenizer)
        sample2_average_embeddings = np.mean(sample2_embeddings, axis = 0)
        serialized_rows = list(set(serialized_rows) - set(sample2_list))
        cosine = CosineSimilarity(sample1_average_embeddings, sample2_average_embeddings)
        sample1_average_embeddings = (sample1_average_embeddings + sample2_average_embeddings) / 2
        #print("Current cosine similarity:", cosine)
        if cosine >= (1 - sim_threshold):
            break
    used_rows = total_rows - len(serialized_rows)               
    # print("Total rows:", total_rows)
    # print("Used rows for serialization:", total_rows - len(serialized_rows))
    return sample1_average_embeddings, total_rows, used_rows

# A function that takes a list of serialized tuples as input and returns embeddings for each tuple as a list with embeddings as value.
def EmbedTuples(tuple_list, model, embedding_type, tokenizer, batch_size = 1000):
    # Initialize an empty dictionary
    final_embedding_list = []

    tuples_batch = []

    if len(tuple_list) > 0:
        # Iterate through sentence list and form batches
        for i in range(0, len(tuple_list)):
            sentence = tuple_list[i]  # For example, Sentence 1, Sentence 2, ...
            tuples_batch.append(sentence)
            # If the batch size is reached or it's the last sentence, embed the batch
            if len(tuples_batch) == batch_size or i == len(tuple_list) - 1:
                if embedding_type == "sentence_bert":
                    tuple_embeddings = model.encode(tuples_batch, convert_to_tensor=True)
                    embeddings_list = tuple_embeddings.cpu().numpy()
                else: #add more for other kinds
                    embeddings_list = encode_finetuned(tuples_batch, model, tokenizer)
                # Add the entries to the dictionary with IDs as the keys and embeddings as the values
                for embedding in embeddings_list:
                    final_embedding_list.append(embedding)
                # Clear the batch for the next set of sentences
                tuples_batch = []
        return final_embedding_list


# A function to load the pretrained model and use it to encode tables.
def encode_finetuned(sentences, model, tokenizer):
    # Tokenize input sentences and convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    encodings = tokenizer(sentences, add_special_tokens = True, truncation = True, padding=True, return_tensors='pt')
    # print("encodings:", encodings)
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    # print("input ids:", input_ids.shape)
    # Generate embeddings for input sentences
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask)
    # print("embeddings tensor:", embeddings.shape)
    # Convert embeddings to numpy array and print
    embeddings = embeddings.cpu().numpy()
    # print("embeddings numpy:", len(embeddings), type(embeddings))
    # print("average embeddings len:", len(np.mean(embeddings, axis = 0)))
    # sys.exit()
    return embeddings

# visualize the results.
def LinePlot(dict_lists, xlabel, ylabel, figname,title):
    # create a list of X-axis values (positions in the list)
    x_values = list(range(1, len(next(iter(dict_lists.values())))+1))
    # print(f'xvalues: {x_values}')
    # create the plot
    for label, values in dict_lists.items():
        plt.plot(x_values, values, label=label)
    
    # set the labels for X and Y axis
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # set the X-axis tick values
    if x_values:
        divisor = max(1,len(x_values)//10)
        # print(f"division:{divisor}")
        if divisor == 0:
            num_ticks = 1
        else:
            num_ticks = len(x_values)//divisor
        step_size = len(x_values)//num_ticks
        # print("step size:", step_size)
        x_ticks = x_values[::step_size]
        # print(f"x ticks: {x_ticks}")
        plt.xticks(x_ticks)
    # plt.ylim(0.1, 1.1)
    # y_ticks = [i/10 for i in range(11)]
    # plt.yticks(y_ticks)
    plt.legend()
    plt.title(title)
    plt.savefig(figname)
    plt.clf()

def read_csv_file(gen_file):
    data = []
    try:
        data = pd.read_csv(gen_file, lineterminator='\n', low_memory=False)
        if data.shape[1] < 2:
            data = pd.read_csv(gen_file, sep='|')
    except:
        try:
            data = pd.read_csv(gen_file, sep='|')
        except:
            with open(gen_file) as curr_csv:
                curr_data = curr_csv.read().splitlines()
                curr_data = [len(row.split('|')) for row in curr_data]
                max_col_num = 0
                if len(curr_data) != 0:
                    max_col_num = max(curr_data)
                try:
                    if max_col_num != 0:
                        df = pd.read_csv(gen_file, sep='|', header=None, names=range(max_col_num), low_memory=False)
                        data = df
                        return data
                    else:
                        df = pd.read_csv(gen_file, lineterminator='\n', low_memory=False)
                        data = df
                        return data
                except:
                    df = pd.read_csv(gen_file, lineterminator='\n', low_memory=False)
                    data = df
                    return data
    return data