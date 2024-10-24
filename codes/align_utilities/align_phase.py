# %%
import random
import glob
import os
import torch
import time
import math
import re
import json
import string
import numpy as np
import tqdm as tqdm
import pandas as pd
import _pickle as cPickle
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import utilities as utl
from transformers import BertTokenizer, BertModel, RobertaTokenizerFast, RobertaModel
import torch, sys
import torch.nn as nn
from torch.nn.parallel import DataParallel
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import itertools
from sklearn import metrics
from glove_embeddings import GloveTransformer
import fasttext_embeddings as ft
import matplotlib.pyplot as plt
from model_classes import BertClassifierPretrained, BertClassifier
random_seed = 42
random.seed(random_seed)

# %%
available_embeddings = ["bert", "bert_serialized", "roberta", "roberta_serialized", "sentence_bert", "sentence_bert_serialized", "glove", "fasttext"]

embedding_type = available_embeddings[3] #change it to fasttext or bert for using the respective embeddings.
use_numeric_columns = True
benchmark_name = "tus_benchmark" 
clustering_metric = "l2" # "cosine"
# %%
print("Embedding type: ", embedding_type)
if embedding_type == "bert" or embedding_type == "bert_serialized":
    model = BertModel.from_pretrained('bert-base-uncased') 
    model = BertClassifierPretrained(model)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vec_length = 768
elif embedding_type == "roberta" or embedding_type == "roberta_serialized":
    model = RobertaModel.from_pretrained("roberta-base")
    model = BertClassifierPretrained(model)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    vec_length = 768
elif embedding_type == "sentence_bert" or embedding_type == "sentence_bert_serialized":
    model = SentenceTransformer('bert-base-uncased') #case insensitive model. BOSTON and boston have the same embedding.
    tokenizer = ""
    vec_length = 768
elif embedding_type == "glove":
    model = GloveTransformer()
    tokenizer = ""
    vec_length = 300
elif embedding_type == "fasttext":
    model = ft.get_embedding_model()
    tokenizer = ""
    vec_length = 300
else:
    print("invalid embedding type")
    sys.exit()



# %%
tfidf_vectorizer = TfidfVectorizer()

# %%
def getColumnType(attribute, column_threshold=0.5, entity_threshold=0.5):
    strAttribute = [item for item in attribute if type(item) == str]
    strAtt = [item for item in strAttribute if not item.isdigit()]
    for i in range(len(strAtt)-1, -1, -1):
        entity = strAtt[i]
        num_count = 0
        for char in entity:
            if char.isdigit():
                num_count += 1
        if num_count/len(entity) > entity_threshold:
            del strAtt[i]            
    if len(strAtt)/len(attribute) > column_threshold:
        return 1
    else:
        return 0

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def findsubsets(s, n):
    return list(itertools.combinations(s, n))

# Function to select tokens based on TF-IDF for each column
def select_values_by_tfidf(values, num_tokens_to_select = 512):
    # tfidf_vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(values)
        tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        sorted_indices = np.argsort(tfidf_scores)[::-1]
        sorted_values = [values[i] for i in sorted_indices]
        return sorted_values[:num_tokens_to_select]
    except ValueError:
        # Handle the case where TF-IDF scores cannot be computed
        print("Returning the same values.")
        if len(values) <= num_tokens_to_select:
            return values
        else:
            return random.sample(values, num_tokens_to_select)
        
def get_glove_embeddings(column_data, sample_size = 50000, sim_threshold = 0.05):     
    sample1_list = random.sample(column_data, min(sample_size, len(column_data)))
    sample1_embeddings = model.transform(sample1_list).reshape(1,-1).flatten()
    column_data = list(set(column_data) - set(sample1_list))
    while(len(column_data) > 0):
        sample2_list = random.sample(column_data, min(sample_size, len(column_data)))
        sample2_embeddings = model.transform(sample2_list).reshape(1,-1).flatten()    
        column_data = list(set(column_data) - set(sample2_list))
        if sample2_embeddings.size == 0:
            continue
        elif sample1_embeddings.size == 0:
            sample1_embeddings = sample2_embeddings
            continue
        else:
            cosine = utl.CosineSimilarity(sample1_embeddings, sample2_embeddings)
            sample1_embeddings = (sample1_embeddings + sample2_embeddings) / 2
            if cosine >= (1 - sim_threshold):
                break
    if sample1_embeddings.size == 0:
        sample1_embeddings = np.random.uniform(-1, 1, 300).astype(np.float32) #glove embedding is 300 vector long             
    return sample1_embeddings

def get_fasttext_embeddings(column_data, sample_size = 50000, sim_threshold = 0.05):     
    sample1_list = random.sample(column_data, min(sample_size, len(column_data)))
    sample1_embeddings = ft.get_fasttext_embeddings(model, sample1_list).reshape(1,-1).flatten() # glove_model.transform(sample1_list).reshape(1,-1)
    column_data = list(set(column_data) - set(sample1_list))
    while(len(column_data) > 0):
        sample2_list = random.sample(column_data, min(sample_size, len(column_data)))
        sample2_embeddings = ft.get_fasttext_embeddings(model, sample2_list).reshape(1,-1).flatten()   
        column_data = list(set(column_data) - set(sample2_list))
        if sample2_embeddings.size == 0:
            continue
        elif sample1_embeddings.size == 0:
            sample1_embeddings = sample2_embeddings
            continue
        else:
            cosine = utl.CosineSimilarity(sample1_embeddings, sample2_embeddings)
            sample1_embeddings = (sample1_embeddings + sample2_embeddings) / 2
            if cosine >= (1 - sim_threshold):
                break
    if sample1_embeddings.size == 0:
        sample1_embeddings = np.random.uniform(-1, 1, 300).astype(np.float32) #glove embedding is 300 vector long     
        # print(sample1_embeddings)        
    return sample1_embeddings
# ft.get_fasttext_embeddings(model, column_data).reshape(1,-1)

def get_sentence_bert_embeddings(column_data, sample_size = 50000, sim_threshold = 0.05): 
    # embedding each column as a table.  
    # print("Column data:", column_data)   
    sample1_embeddings = utl.EmbedTable(column_data, embedding_type="sentence_bert", model=model, tokenizer=tokenizer)[0]
    
    # print("type: ", type(sample1_embeddings)) #.shape()) #.shape())
    # print(sample1_embeddings)
    if np.isnan(sample1_embeddings).any():
        # print("sample embedding is nan")
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32) #sbert embedding is 768 vector long     
    # print("len:", len(sample1_embeddings))
    return sample1_embeddings

def get_bert_embeddings(column_data, sample_size = 50000, sim_threshold = 0.05, embedding_type = "bert"): 
    # embedding each column as a table.  
    # print("Column data:", column_data)   
    sample1_embeddings = utl.EmbedTable(column_data, embedding_type=embedding_type, model=model, tokenizer=tokenizer)[0]
    
    # print("type: ", type(sample1_embeddings)) #.shape()) #.shape())
    # print(sample1_embeddings)
    if np.isnan(sample1_embeddings).any():
        # print("sample embedding is nan")
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32) #sbert embedding is 768 vector long     
    # print("len:", len(sample1_embeddings))
    return sample1_embeddings


def get_sentence_bert_embeddings_serialize(column_data, sample_size = 512, sim_threshold = 0.05): 
    selected_tokens = select_values_by_tfidf(column_data, num_tokens_to_select = sample_size)
    selected_tokens = ' '.join(selected_tokens)
    sample1_embeddings = utl.EmbedTable([selected_tokens], embedding_type="sentence_bert", model=model, tokenizer=tokenizer)[0]
    # print("sample1 embeddings: ", sample1_embeddings)
    if np.isnan(sample1_embeddings).any():
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32) #sbert embedding is 768 vector long     
    return sample1_embeddings

def get_bert_embeddings_serialize(column_data, sample_size = 512, sim_threshold = 0.05, embedding_type = "bert"): 
    selected_tokens = select_values_by_tfidf(column_data, num_tokens_to_select = sample_size)
    selected_tokens = ' '.join(selected_tokens)
    sample1_embeddings = utl.EmbedTable([selected_tokens], embedding_type=embedding_type, model=model, tokenizer=tokenizer)[0]
    # print("sample1 embeddings: ", sample1_embeddings)
    if np.isnan(sample1_embeddings).any():
        sample1_embeddings = np.random.uniform(-1, 1, 768).astype(np.float32) #sbert embedding is 768 vector long     
    return sample1_embeddings


# collect the columns within the whole data lake for lsh ensemble.
def compute_embeddings(table_path_list, embedding_type, max_col_size = 10000, use_numeric_columns = True):
    computed_embeddings = {}
    using_tables = 0
    zero_columns = 0
    numeric_columns = 0
    # print(table_path_list)
    for file in table_path_list:
        #try:
        df = utl.read_csv_file(file)
        if len(df) < 3:
            continue
        else:
            using_tables += 1
        # df = pd.read_csv(file, encoding = "latin-1", on_bad_lines="skip", lineterminator="\n")
        table_name = file.rsplit(os.sep,1)[-1]
        for idx, column in enumerate(df.columns):
            column_data = list(set(df[column].map(str)))
            if use_numeric_columns == False and getColumnType(column_data) == 0:
                numeric_columns += 1
                continue
            column_data = random.sample(column_data, min(len(column_data), max_col_size))
            all_text = ' '.join(column_data)
            # .join(map(str.lower, original_list))
            all_text = re.sub(r'\([^)]*\)', '', all_text)
            column_data = list(set(re.sub(r"[^a-z0-9]+", " ", all_text.lower()).split()))
            if len(column_data) == 0:
                zero_columns += 1
                continue
            # print("Total column values:", len(column_data))                
            if embedding_type == "glove":
                this_embedding = get_glove_embeddings(column_data)
            elif embedding_type == "fasttext":
                this_embedding = get_fasttext_embeddings(column_data)
            elif embedding_type == "sentence_bert":
                this_embedding = get_sentence_bert_embeddings(column_data)
            elif embedding_type == "sentence_bert_serialized":
                this_embedding = get_sentence_bert_embeddings_serialize(column_data)
            elif embedding_type == "bert" or embedding_type == "roberta":
                this_embedding = get_bert_embeddings(column_data, embedding_type = embedding_type)
            elif embedding_type == "bert_serialized" or embedding_type == "roberta_serialized":
                this_embedding = get_bert_embeddings_serialize(column_data, embedding_type = embedding_type)
            computed_embeddings[(table_name, column)] = this_embedding
    print(f"Embedded {using_tables} table(s), {len(computed_embeddings)} column(s). Zero column(s): {zero_columns}. Numeric Column(s): {numeric_columns}")
    return computed_embeddings

def print_metrics(precision, recall, f_measure):
    total_precision = 0
    total_recall = 0
    total_f_measure = 0
    average_precision = 0
    average_recall = 0
    average_f_measure = 0

    for item in precision:
        total_precision += precision[item]
    for item in recall:
        total_recall += recall[item]
    for item in f_measure:
        total_f_measure += f_measure[item]

    average_precision = total_precision / len(precision)
    average_recall = total_recall / len(recall)
    average_f_measure = total_f_measure / len(f_measure)
    print("Average precision:", average_precision)
    print("Average recall", average_recall)
    print("Average f measure", average_f_measure)

def plot_accuracy_range(original_dict, embedding_type, benchmark_name, metric = "F1-score", range_val = 0.1, save = False, save_folder = r"plots_align"):
    save_name = f"{save_folder + os.sep}_{metric}_{benchmark_name}_{embedding_type}"
    range_count_dict = {}

    # Create keys with ranges from 0 to 1 with 0.1 difference and set values to 0
    for i in range(0,10):
        start_range = i / 10.0
        end_range = (i + 1) / 10.0
        key = f'{start_range:.1f}-{end_range:.1f}'
        range_count_dict[key] = 0
    # Iterate through the original dictionary values
    for value in original_dict.values():
        # Determine the range for the current value
        for key_range in range_count_dict:
            range_start, range_end = map(float, key_range.split('-'))
            if range_start <= value < range_end:
                # Increment the count for the corresponding range
                range_count_dict[key_range] += 1
                break  # Exit the loop once the range is found

    # Extract data for plotting
    ranges = list(range_count_dict.keys())
    counts = list(range_count_dict.values())

    plt.figure(figsize=(10, 6))  # Adjust the width and height as needed


    # Plotting the bar graph
    plt.bar(ranges, counts, color='blue')

    # Adding labels and title
    plt.xlabel('Ranges')
    plt.ylabel('Number of Tables')
    plt.title(f'{embedding_type} F1-score in {benchmark_name}')

    # Adding text on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    if save == True:
        plt.savefig(save_name)
    else:
        plt.show()
    plt.clf()


# %%
# Opening JSON file


final_precision = {}
final_recall = {}
final_f_measure = {}
final_query_precision = {}
final_query_recall = {}
final_query_f_measure = {}


dl_table_folder = r"data" + os.sep + benchmark_name + os.sep + "datalake"
query_table_folder = r"data" + os.sep + benchmark_name + os.sep + "query"
groundtruth_file = r"groundtruth" + os.sep + benchmark_name + "_union_groundtruth.pickle"  # key is query name and value is the list of unionable tables.

query_tables = glob.glob(query_table_folder + os.sep + "*.csv")
groundtruth = utl.loadDictionaryFromPickleFile(groundtruth_file)

align_plot_folder = r"plots_align"

# %%
start_time = time.time_ns()
used_queries = 0
# evaluation of align phase.
for query_table in query_tables:
    query_table_name = query_table.rsplit(os.sep, 1)[-1]
    print(query_table_name)
    column_embeddings = []
    track_tables = {}
    track_columns = {}
    record_same_cluster = {}
    query_column_ids = set()
    i = 0
    if query_table_name not in groundtruth:
        continue
    else:
        unionable_tables = groundtruth[query_table_name]
        # get embeddings for query table columns.
        query_embeddings = compute_embeddings([query_table], embedding_type, use_numeric_columns=use_numeric_columns)
        if len(query_embeddings) == 0:
            print("Not enough rows. Ignoring this query table.")
            continue
        used_queries += 1
        unionable_table_path = [dl_table_folder + os.sep + tab for tab in unionable_tables if tab != query_table_name]
        unionable_table_path = [path for path in unionable_table_path if os.path.exists(path)]
        if benchmark_name == "tus_benchmark":
            unionable_table_path = random.sample(unionable_table_path, min(10, len(unionable_table_path))) 
        dl_embeddings = compute_embeddings(unionable_table_path, embedding_type, use_numeric_columns= use_numeric_columns)
        if len(dl_embeddings) == 0:
            print("Not enough rows in any data lake tables. Ignoring this cluster.")
        for column in query_embeddings:
            column_embeddings.append(query_embeddings[column])
            track_columns[column] = i
            if column[0] not in track_tables:
                track_tables[column[0]] = {i}
            else:
                track_tables[column[0]].add(i)
            if column[1] not in record_same_cluster:
                record_same_cluster[column[1]] =  {i}
            else:
                record_same_cluster[column[1]].add(i)
            query_column_ids.add(i)
            i += 1
        for column in dl_embeddings:
            column_embeddings.append(dl_embeddings[column])
            track_columns[column] = i
            if column[0] not in track_tables:
                track_tables[column[0]] = {i}
            else:
                track_tables[column[0]].add(i)
            if column[1] not in record_same_cluster:
                record_same_cluster[column[1]] =  {i}
            else:
                record_same_cluster[column[1]].add(i)
            i += 1
            
        all_true_edges = set() 
        all_true_query_edges = set()
        for col_index_set in record_same_cluster:
            set1 = record_same_cluster[col_index_set]
            set2 = record_same_cluster[col_index_set]
            current_true_edges = set()
            current_true_query_edges = set()
            for s1 in set1:
                for s2 in set2:
                    current_true_edges.add(tuple(sorted((s1,s2))))
                    if s1 in query_column_ids or s2 in query_column_ids:
                        current_true_query_edges.add(tuple(sorted((s1,s2))))
            all_true_edges = all_true_edges.union(current_true_edges)  
            all_true_query_edges = all_true_query_edges.union(current_true_query_edges) 
        column_embeddings = list(column_embeddings)
        x = np.array(column_embeddings)
        
        zero_positions = set()
        for table in track_tables:
            indices = track_tables[table]
            all_combinations = findsubsets(indices, 2)
            for each in all_combinations:
                zero_positions.add(each)
        
        arr = np.zeros((len(track_columns),len(track_columns)))
        for i in range(0, len(track_columns)-1):
            for j in range(i+1, len(track_columns)):
                #print(i, j)
                if (i, j) not in zero_positions and (j, i) not in zero_positions and i !=j:
                    arr[i][j] = 1
                    arr[j][i] = 1
        # convert to sparse matrix representation 
        s = csr_matrix(arr)  

        all_distance = {}
        all_labels = {}
        record_current_precision = {}
        record_current_recall = {}
        record_current_f_measure = {}
        record_current_query_precision = {}
        record_current_query_recall = {}
        record_current_query_f_measure = {}
        min_k = len(query_embeddings)
        max_k = 0
        record_result_edges = {}
        record_result_query_edges = {}
        
        for item in track_tables:
            #print(item, len(track_tables[item]))
            if len(track_tables[item])> min_k:
                min_k = len(track_tables[item])
            max_k += len(track_tables[item])
        
        
        for i in range(min_k, min(max_k, max_k)):
            #clusters = KMeans(n_clusters=14).fit(x)
            clusters = AgglomerativeClustering(n_clusters=i, metric=clustering_metric,
                        compute_distances = True , linkage='average', connectivity = s)
            clusters.fit_predict(x)
            labels = (clusters.labels_) #.tolist()
            all_labels[i]= labels.tolist()
            all_distance[i] = metrics.silhouette_score(x, labels)
            result_dict = {}
            wrong_results = set()
            for (col_index, label) in enumerate(all_labels[i]):
                if label in result_dict:
                    result_dict[label].add(col_index)
                else:
                    result_dict[label] = {col_index}
            
            
            all_result_edges = set() 
            all_result_query_edges = set()
            for col_index_set in result_dict:
                set1 = result_dict[col_index_set]
                set2 = result_dict[col_index_set]
                current_result_edges = set()
                current_result_query_edges = set()
                for s1 in set1:
                    for s2 in set2:
                        current_result_edges.add(tuple(sorted((s1,s2))))
                        if s1 in query_column_ids or s2 in query_column_ids:
                            current_result_query_edges.add(tuple(sorted((s1,s2))))
                all_result_edges = all_result_edges.union(current_result_edges)
                all_result_query_edges = all_result_query_edges.union(current_result_query_edges)
            current_true_positive = len(all_true_edges.intersection(all_result_edges))
            current_precision = current_true_positive/len(all_result_edges)
            current_recall = current_true_positive/len(all_true_edges)

            current_query_true_positive = len(all_true_query_edges.intersection(all_result_query_edges))
            current_query_precision = current_query_true_positive/len(all_result_query_edges)
            current_query_recall = current_query_true_positive/len(all_true_query_edges)
            

            record_current_precision[i] = current_precision
            record_current_recall[i] = current_recall
            record_current_f_measure[i] = 0
            if (current_precision + current_recall) > 0:
                record_current_f_measure[i] = (2 * current_precision * current_recall)/ (current_precision + current_recall)              
            record_result_edges[i] = all_result_edges

            record_current_query_precision[i] = current_query_precision
            record_current_query_recall[i] = current_query_recall
            record_current_query_f_measure[i] = 0
            if (current_query_precision + current_query_recall) > 0:
                record_current_query_f_measure[i] = (2 * current_query_precision * current_query_recall)/ (current_query_precision + current_query_recall)              
            record_result_query_edges[i] = all_result_query_edges
        
        distance_list = all_distance.items()
        distance_list = sorted(distance_list) 
        x, y = zip(*distance_list)
        algorithm_k = max(all_distance, key=all_distance. get) 
        final_precision[query_table_name] = record_current_precision[algorithm_k]
        final_recall[query_table_name] = record_current_recall[algorithm_k]
        final_f_measure[query_table_name] = record_current_f_measure[algorithm_k]
        final_query_precision[query_table_name] = record_current_query_precision[algorithm_k]
        final_query_recall[query_table_name] = record_current_query_recall[algorithm_k]
        final_query_f_measure[query_table_name] = record_current_query_f_measure[algorithm_k]
        overlapping = 1
        continue
        plt.plot(x, y)
        plt.title(query_table_name)
        plt.xlabel("number of clusters")
        plt.ylabel("silhouette score")
        plt.axvline(x = len(record_same_cluster), color = 'red', linestyle = "dashed", label = 'groundtruth k', alpha=overlapping, lw=3)
        plt.axvline(x = algorithm_k, linestyle = "dotted", color = 'green', label = 'algorithm k', alpha=overlapping, lw=3)
        plt.axvline(x = min_k, color = 'black', label = 'min k')
        #plt.axvline(x = max_k, color = 'black', label = 'max k')
        #plt.legend(bbox_to_anchor = (1.0, 1), loc = 'lower right', borderaxespad=3)
        plt.show()
        # print(query_embeddings)
end_time = time.time_ns()

# %%
total_time = int(end_time - start_time)/ 10 **9
print("Method:", embedding_type, "|| Benchmark: ", benchmark_name)
print("-------------------------------------")
print("Total time:", total_time)
print("-------------------------------------")
print("Results:")
print_metrics(final_precision, final_recall, final_f_measure)
print("-------------------------------------")
print("Average time per query = ", total_time / used_queries)
