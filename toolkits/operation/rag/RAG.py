import os
import torch
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from data_utils import load_jsonlines, save_jsonlines, load_document_embeddings, save_document_embeddings
from tqdm import tqdm


    
def get_args():
    parser = argparse.ArgumentParser(description='Process input and output paths.')
    parser.add_argument('--input_file_path', type=str, default="input.jsonl", help='Input file path')
    parser.add_argument('--output_path', type=str, default='output.jsonl', help='Output file path')
    parser.add_argument('--db_embedding_path', type=str, default="wiki_emb.npy", help='Database embedding path')
    parser.add_argument('--db_file_path', type=str, default="enwiki_processed.jsonl", help='Database file path')
    parser.add_argument('--top_k', type=int, default=2, help='Top K documents to retrieve')
    return parser.parse_args()


def encode_documents(model, texts, batch_size=50):
    return model.encode(texts, convert_to_tensor=True, device="cuda", show_progress_bar=True, batch_size=batch_size)


def batch_search(query_embeddings, document_embeddings, top_k, chunk_size):
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)
    top_k_indices, top_k_scores = [], []

    for i in range(0, document_embeddings.size(0), chunk_size):
        chunk = document_embeddings[i:i + chunk_size]
        cosine_similarities = torch.mm(query_embeddings, chunk.t())
        top_k_scores_chunk, top_k_indices_chunk = torch.topk(cosine_similarities, k=top_k, dim=1)
        if i == 0:
            top_k_scores, top_k_indices = top_k_scores_chunk, top_k_indices_chunk
        else:
            combined_scores, combined_indices = torch.topk(torch.cat((top_k_scores, cosine_similarities), dim=1), k=top_k, dim=1)
            top_k_scores = combined_scores
            top_k_indices = torch.where(combined_indices < top_k, top_k_indices, combined_indices - top_k + i)

    return top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()


def processes(args):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to('cuda')
    documents = load_jsonlines(args.db_file_path)
   
    if not os.path.exists(args.db_embedding_path):
        document_embeddings = encode_documents(model, documents)
        save_document_embeddings(document_embeddings.cpu().numpy(), args.db_embedding_path)
    else:
        document_embeddings = load_document_embeddings(args.db_embedding_path)
        


    data = load_jsonlines(args.input_file_path)
    query_embeddings = encode_documents(model, [d['instruction'] for d in data])

    top_k_indices, top_k_scores = [], []
    for i in tqdm(range(0, query_embeddings.size(0), 2000), desc='searching'):
        chunk = query_embeddings[i:i + 2000]
        top_k_index, top_k_score = batch_search(chunk, document_embeddings, args.top_k, 2000)
        top_k_indices.append(top_k_index)
        top_k_scores.append(top_k_score)

    top_k_indices = np.concatenate(top_k_indices, axis=0)

    for query_idx, query in enumerate(data):
        for i in range(args.top_k):
            data[query_idx][f'doc{i+1}'] = documents[top_k_indices[query_idx][i]]

    save_jsonlines(data, args.output_path)


if __name__ == "__main__":
    args = get_args()
    processes(args)
    
    
    
    
##########################
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from data_utils import load_jsonlines, save_jsonlines, load_document_embeddings, save_document_embeddings
from tqdm import tqdm
import fire
from config import Config
from typing import List, Dict


def encode_documents(model, texts, batch_size=50):
    return model.encode(texts, convert_to_tensor=True, device="cuda", show_progress_bar=True, batch_size=batch_size)

def batch_search(query_embeddings, document_embeddings, top_k, chunk_size):
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=1)
    top_k_indices, top_k_scores = [], []

    for i in range(0, document_embeddings.size(0), chunk_size):
        chunk = document_embeddings[i:i + chunk_size]
        cosine_similarities = torch.mm(query_embeddings, chunk.t())
        top_k_scores_chunk, top_k_indices_chunk = torch.topk(cosine_similarities, k=top_k, dim=1)
        if i == 0:
            top_k_scores, top_k_indices = top_k_scores_chunk, top_k_indices_chunk
        else:
            combined_scores, combined_indices = torch.topk(torch.cat((top_k_scores, cosine_similarities), dim=1), k=top_k, dim=1)
            top_k_scores = combined_scores
            top_k_indices = torch.where(combined_indices < top_k, top_k_indices, combined_indices - top_k + i)

    return top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()



def find_relevant_docs(data: List[Dict[str, str]], config: Config, use_old = True) -> List[Dict[str, str]]:

    model = SentenceTransformer(config.encode_model_pth).to('cuda')
    db_file_path = config.old_file_path if use_old else config.new_file_path
    db_embedding_path = config.embedding_pth_old if use_old else config.embedding_pth_new   
    documents = [item['text'] for item in load_jsonlines(db_file_path)]
    if not os.path.exists(db_embedding_path):
        document_embeddings = encode_documents(model, documents)
        save_document_embeddings(document_embeddings.cpu().numpy(), db_embedding_path)
    else:
        document_embeddings = load_document_embeddings(db_embedding_path)

    query_embeddings = encode_documents(model, [d['instruction'] for d in data])

    top_k_indices, top_k_scores = [], []
    for i in tqdm(range(0, query_embeddings.size(0), 2000), desc='searching'):
        chunk = query_embeddings[i:i + 2000]
        top_k_index, top_k_score = batch_search(chunk, document_embeddings, config.top_K, 2000)
        top_k_indices.append(top_k_index)
        top_k_scores.append(top_k_score)

    top_k_indices = np.concatenate(top_k_indices, axis=0)

    for query_idx, query in enumerate(data):
        data[query_idx]['additional_docs'] = [documents[top_k_indices[query_idx][i]] for i in range(config.top_K)]
        
        
if __name__ == "__main__":
    # fire.Fire(processes)
    data = load_jsonlines("/home/admin/FANNO/Fanno/experiment/experiment1_20000.jsonl")
    find_relevant_docs(data, Config(), use_old=False)
    save_jsonlines(data, "/home/admin/FANNO/Fanno/experiment/experiment1_20000_with_doc.jsonl")
    
    

    
    
    