import pandas as pd
import os, glob, pickle
import signal, json, requests
import _pickle as cPickle
import bz2, sys
from tqdm import tqdm
from fasttext import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.request import urlopen
import gzip, shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

##################################################################################################
#   Fasttext code starts here. Todo : make a class and move the code for fasttext to a class.
##################################################################################################


def download_fasttext_model(model_file_name: str, chunk_size: int = 2 ** 13):
    FASTTEXTURL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/"
    url = FASTTEXTURL + model_file_name
    print("Downloading %s" % url)
    response = urlopen(url)

    downloaded = 0
    write_file_name = model_file_name
    download_file_name = write_file_name + ".part"
    with open(download_file_name, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            downloaded += len(chunk)
            if not chunk:
                break
            f.write(chunk)
            # print("{} downloaded ...".format(downloaded))
    os.rename(download_file_name, write_file_name)

def download_model(if_exists: str = "strict"):
    """
    Download the pre-trained model file.
    Parameters
    ----------
    if_exists : str
        Supported values:
            - *ignore*: The model will not be downloaded
            - *strict*: This is the defaul. The model will be downloaded only if it does not exist at the *cache_dir*.
            - *overwrite*: The model will be downloaded even if it already exists at the *cache_dir*.
    Returns
    -------
    """
    cache_dir = None,
    # cache_dir  =  cache_dir if cache_dir is not None and os.path.isdir(cache_dir) else None
    base_file_name = "cc.%s.300.bin" % "en"
    file_name = base_file_name
    gz_file_name = "%s.gz" % base_file_name

    if os.path.isfile(file_name):
        if if_exists == "ignore":
            return file_name
        elif if_exists == "strict":
            print("File exists. Use --overwrite to download anyway.")
            return file_name
        elif if_exists == "overwrite":
            pass

    absolute_gz_file_name = gz_file_name
    if not os.path.isfile(absolute_gz_file_name):
        download_fasttext_model(gz_file_name)

    with gzip.open(absolute_gz_file_name, "rb") as f:
        with open(file_name, "wb") as f_out:
            shutil.copyfileobj(f, f_out)

    """Cleanup"""
    if os.path.isfile(absolute_gz_file_name):
        os.remove(absolute_gz_file_name)
    return file_name

def get_embedding_model(overwrite: bool = False):
    """
    Download, if not exists, and load the pretrained FastText embedding model in the working directory.
    Note that the default gzipped English Common Crawl FastText model has 4.2 GB
    and its unzipped version has 6.7 GB.
    Parameters
    ----------
    overwrite : bool
        If True overwrites the model if exists.
    Returns
    -------
    """
    if_exists = "strict" if not overwrite else "overwrite"

    model_file = download_model(if_exists=if_exists)
    embedding_model = load_model(model_file)
    return embedding_model

def select_tokens(value_list):
    # Create the TfidfVectorizer with stopwords removed
    vectorizer = TfidfVectorizer(stop_words='english')
    # Get the list of stopwords from the TfidfVectorizer
    stopwords = set(vectorizer.get_stop_words())

    selected_tokens_list = []
    for val in value_list:

        # Tokenize the sentence into individual words
        tokens = val.split()

        # Remove stopwords
        tokens_without_stopwords = [token.lower() for token in tokens if token.lower() not in stopwords]

        # Combine the remaining tokens into a single sentence representation
        filtered_value = " ".join(tokens_without_stopwords)
        filtered_value = filtered_value.strip()
        if len(filtered_value) > 0:
            selected_tokens_list.append(filtered_value)
    return selected_tokens_list

def get_fasttext_embeddings(model, value_list):
    # Get embeddings for representative tokens using all column values
    representative_token = select_tokens(value_list)
    representative_token_embeddings = [model.get_word_vector(token) for token in representative_token]
    # Calculate the average embedding
    if len(representative_token_embeddings) == 0:
            return np.empty(0)
    combined_embedding = np.mean(representative_token_embeddings, axis=0)
    return combined_embedding

##################################################################
# Fasttext code ends here.
##################################################################



