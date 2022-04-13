from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError, RequestError
import torch
from transformers import T5Tokenizer, T5Model
import json
TOKEN_DIR='./tokenizer'
MODEL_DIR='./model'

# Number of results
SEARCH_SIZE = 10

def get_emb(inputs_list):
    tokenizer = T5Tokenizer.from_pretrained(TOKEN_DIR)
    model = T5Model.from_pretrained(MODEL_DIR)
    inputs = tokenizer.batch_encode_plus(inputs_list, max_length = 100, padding='max_length', truncation=True, return_tensors="pt")
    outputs = model(input_ids=inputs['input_ids'], decoder_input_ids=inputs['input_ids'])
    last_hidden_states = torch.mean(outputs[0], dim=1)
    return last_hidden_states.tolist()

client = Elasticsearch()

# Insert search term into query variable 
query = "fun and games" 
query_vector = get_emb(inputs_list=[query])[0]

script_query = {
    "script_score": {
        "query": {"match_all": {}},
        "script": {
            "source": "cosineSimilarity(params.query_vector, doc['desc_vector']) + 1.0",
            "params": {"query_vector": query_vector}
        }
    }
}

try:
    response = client.search(
         index="apple",  # Name of index
         body={
             "size": SEARCH_SIZE,
             "query": script_query,
             "_source": {"includes": ["app", "app_desc"]}
         }
     )

    json_response = json.dumps(response, indent=2)
    print(json_response)

except ConnectionError:
    print("[WARNING] Docker isn't up and running!")
except NotFoundError:
    print("[WARNING] No such index!")
except RequestError as e:
    print(e.info)