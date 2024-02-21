#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer, util
import os

def semantic_search(corpus_path, sentence):
    filelist = []
    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)

        # Check if the current item is a file (not a subdirectory)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    filelist.append(file_content)
            except UnicodeDecodeError as e:
                print(f"Error reading {file_path}: {str(e)}")
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    query_embedding = model.encode(sentence)
    passage_embedding = model.encode(filelist)
    print(util.dot_score(query_embedding, passage_embedding))
    highest = util.dot_score(query_embedding, passage_embedding).argmax()
    return (filelist[highest])
