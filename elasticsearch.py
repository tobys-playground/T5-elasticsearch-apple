import pandas as pd
import argparse
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import torch
from transformers import T5Tokenizer, T5Model
TOKEN_DIR='./tokenizer'
MODEL_DIR='./model'

def get_emb(inputs_list):
    tokenizer = T5Tokenizer.from_pretrained(TOKEN_DIR)
    model = T5Model.from_pretrained(MODEL_DIR)
    inputs = tokenizer.batch_encode_plus(inputs_list, max_length = 100, padding='max_length', truncation=True, return_tensors="pt")
    outputs = model(input_ids=inputs['input_ids'], decoder_input_ids=inputs['input_ids'])
    last_hidden_states = torch.mean(outputs[0], dim=1)
    return last_hidden_states.tolist()

def create_document(doc, emb, index_name):
    return {
        '_op_type': 'index',
        '_index': index_name,
        'app': doc['app'],
        'app_desc': doc['app_desc'],
        'desc_vector': emb
    }

def load_csv_dataset(path):
    docs = []
    df = pd.read_csv(path)
    df = df[:100] #select only 100 rows for simplicity
    for row in df.iterrows():
        series = row[1]
        doc = {
            'app': series.track_name,
            'app_desc': series.app_desc
        }
        docs.append(doc)
    return docs

def bulk_predict(docs, batch_size=256):
    """Predict T5 embeddings."""
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
        embeddings = get_emb(inputs_list=[doc['app_desc'] for doc in batch_docs])
        for emb in embeddings:
            yield emb

def load_docs(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def main(args):
    # Create documents and store in documents.jsonl file
    print("Loading data from csv file....")
    docs = load_csv_dataset(args.csv)
    print("Creating documents...")
    with open(args.data, 'w') as f:
        for doc, emb in zip(docs, bulk_predict(docs)):
            d = create_document(doc, emb, args.index_name)
            f.write(json.dumps(d) + '\n')

    # Create index
    print("Creating index...")
    client = Elasticsearch()
    #client.indices.delete(index=args.index_name, ignore=[404])
    with open(args.index_config) as index_config:
        source = index_config.read().strip()
        client.indices.create(index=args.index_name, body=source)

    # Index documents
    print("Indexing documents...")
    client = Elasticsearch()
    docs = load_docs(args.data)
    bulk(client, docs)
    print("Check Kibana for documents under created index...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating Elasticsearch documents')
    parser.add_argument('--index_config', default='index.json', help='Elasticsearch index config file')
    parser.add_argument('--index_name', default='apple', help='Index name in Elasticsearch')
    parser.add_argument('--csv', default='apple_store_apps_desc.csv', help='Path of input csv file')
    parser.add_argument('--data', default='documents.jsonl', help='File that contains created documents')
    args = parser.parse_args()
    main(args)