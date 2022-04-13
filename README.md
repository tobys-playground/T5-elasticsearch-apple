## An Enhanced Information Retrieval System (Elasticsearch) for the Apple App Store

**Aim**: To implement an **Information Retrieval system** enhanced by a **pre-trained T5 model** for the **Apple App Store** using **Elasticsearch and Kibana Helm charts** on **Kubernetes** (Minikube)

## Requirements

1) Docker
2) Minikube
3) Kubectl
4) Helm
5) Packages in requirements.txt

## Set-up

1) Start Minikube cluster: `minikube start`
2) Add Elastic Repo: `Helm repo add elastic https://Helm.elastic.co`
3) Install Elasticsearch Helm chart: `Helm install elasticsearch elastic/elasticsearch -f ./values.yaml`
4) Port-forward Elasticsearch to local port 9200: `kubectl port-forward svc/elasticsearch-master 9200`
5) Install Kibana Helm chart: `Helm install kibana elastic/kibana`
6) Port-forward Kibana to local port 5601: `kubectl port-forward deployment/kibana-kibana 5601`
7) See Minikube details: `minikube dashboard`
8) Open Kibana at 'localhost:5601'
9) **Download the T5 tokenizer and model from HuggingFace**

## Main files

1) **apple_store_apps_desc.csv** - Useful fields: app name (**track_name**) and app description (**app_desc**). This dataset was taken from [Kaggle](https://www.kaggle.com/datasets/ramamet4/app-store-apple-data-set-10k-apps?resource=download)
2) **index.json** - Index configuration and mappings (fields are **'app', 'app_desc', and 'desc_vector'**, which are T5 embeddings of app_desc)
3) **elasticsearch.py** - For 1) generating pre-trained T5 embeddings, documents, and index, and 2) indexing documents in Elasticsearch
4) **test_elasticsearch.py** - To test if Elasticsearch could return results with search queries

## Steps

1) **Documents** will be created based on the **mappings in index.json** by the create_document function in elasticsearch.py, and stored in **documents.jsonl**
2) **The index 'apple'** will be created in Elasticsearch by the main function in elasticsearch.py
3) **Documents will be indexed under 'apple'** by the main function in elasticsearch.py
4) **Verify that the index and documents have been added in Elasticsearch via Kibana**. For each app, its name, description, and the T5 embeddings of its description should be present:

![image](https://user-images.githubusercontent.com/81354022/163163880-134b94a4-de61-4833-88f8-865221c9bd1e.png)
![image](https://user-images.githubusercontent.com/81354022/163164212-a842b8cf-b958-46b3-bc96-ab621ef2672d.png)

5) **Enter a query in test_elasticsearch.py**. The embeddings of this query will be generated and compared to the embeddings stored in Elasticsearch. The results most similar to the query's embeddings (using **Cosine Similarity**) will be returned:

![image](https://user-images.githubusercontent.com/81354022/163162327-8721e18f-31cb-4468-8ca0-c22c4b215ec5.png)

## References

1) https://github.com/kelvin-jose/elasticbert
2) https://github.com/QiuruiChen/T5Elasticsearch
