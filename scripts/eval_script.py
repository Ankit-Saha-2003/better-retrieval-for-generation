import json
import faiss_database
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import torch
from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
import pandas as pd
import numpy as np
from llama_index.embeddings.adapter.utils import TwoLayerNN
from llama_index.core.embeddings import resolve_embed_model
from metrics import contextual_precision, contextual_recall, contextual_relevancy
from tqdm import tqdm

verbose = True

train_df = pd.read_parquet('train-00001-of-00007.parquet')

train_queries = dict()
corpus = dict()
train_relevant_docs = dict()

count = 0

for index, row in train_df.iterrows():
    
    train_queries[f'{index}'] = row['query']
        
    for corpus_index, passage in enumerate(row['passages']['passage_text']):

        corpus[f'{index}.{corpus_index}'] = passage
    
    train_relevant_docs[f'{index}'] = [f'{index}.{i}' for i in range(len(row['passages']['passage_text']))] 
    
    
train_dataset = EmbeddingQAFinetuneDataset(
    queries = train_queries, corpus = corpus, relevant_docs = train_relevant_docs
)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
base_embed_model = resolve_embed_model(f"local:{model_name}")

# adapter_model = TwoLayerNN(
#     in_features=384,
#     hidden_features=784,
#     out_features=384,
#     bias=True,
#     add_residual=True
# )
model_path = "chks/llama_index/all-MiniLM-L6-v2-finetuned-TwoLayerNN-v2"
config_path = f"{model_path}/config.json"
model_weights_path = f"{model_path}/pytorch_model.bin"

# Load the config
with open(config_path, "r") as f:
    config = json.load(f)

# Reconstruct the TwoLayerNN adapter model
adapter_model = TwoLayerNN(
    in_features=config["in_features"],
    hidden_features=config["hidden_features"],
    out_features=config["out_features"],
    bias=config["bias"],
    activation_fn_str=config["activation_fn_str"],
    add_residual=config["add_residual"]
)


finetune_engine = EmbeddingAdapterFinetuneEngine(
    train_dataset,
    base_embed_model,
    model_output_path="chks/llama_index/all-MiniLM-L6-v2-finetuned-TwoLayerNN-v3",
    adapter_model=adapter_model,
    epochs=10,
    verbose=False,
    device="cuda",
    batch_size = 64
)

if verbose:
    
    print("---------------FineTuning start---------------------------")
    
finetune_engine.finetune()

if verbose:
    
    print("--------------------------Model Finetuned!-----------------------")

# Paths to the saved files

if verbose:
    
    print("--------------------Load finetuned model------------------------")
    
model_path = "chks/llama_index/all-MiniLM-L6-v2-finetuned-TwoLayerNN-v3"
config_path = f"{model_path}/config.json"
model_weights_path = f"{model_path}/pytorch_model.bin"

# Load the config
with open(config_path, "r") as f:
    config = json.load(f)

# Reconstruct the TwoLayerNN adapter model
adapter_model = TwoLayerNN(
    in_features=config["in_features"],
    hidden_features=config["hidden_features"],
    out_features=config["out_features"],
    bias=config["bias"],
    activation_fn_str=config["activation_fn_str"],
    add_residual=config["add_residual"]
)

# Load the adapter model's weights
adapter_model.load_state_dict(torch.load(model_weights_path,  map_location=torch.device('cpu')))

# Load the base SentenceTransformer model
base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device = 'cpu')

test_df = pd.read_parquet('test-00000-of-00001.parquet')

index = faiss_database.IndexFlatIP(384)
# index_gpu = faiss.index_cpu_to_all_gpus(index)

def embedding_function(text):
    # inputs = tokenizer(text, return_tensors="pt")
    # outputs = model(**inputs)
    # return outputs.last_hidden_state.detach().numpy()[0, 0]
    embeddings = base_model.encode(text)
    embeddings = adapter_model.forward(torch.tensor(embeddings, dtype = torch.float32, requires_grad=False)).detach()
    return embeddings / np.linalg.norm(embeddings)


vector_store = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

unique_passages = set()
index_to_docstore_id = {}
document_counter = 0
batch_documents = []

if verbose:
    
    print("-----------------------------Starting making vector index---------------------------------")
    
# Loop over passages using tqdm for progress tracking
for passages in tqdm(test_df['passages']):
  for passage in passages['passage_text']:

    passage_text = passage# Assuming the structure of the dict is correct
    tokens = len(base_model.tokenizer(passage)['input_ids'])

    if tokens <= 256 and passage_text not in unique_passages:
      unique_passages.add(passage_text)  # Add to set
      document = Document(page_content=passage_text)
      batch_documents.append(document)

      index_to_docstore_id[document_counter] = passage_text  # Store mapping
      document_counter += 1

      # Process and add documents in batches
      if len(batch_documents) >= 10000:  # Adjust batch size as necessary
        vector_store.add_documents(documents=batch_documents)
        batch_documents = []  # Clear the batch after adding
          # print('----')

# Add any remaining documents
if batch_documents:
    vector_store.add_documents(documents=batch_documents)
  
import pickle

faiss_database.write_index(index, "finetune_llamindex_v3_faiss_index.bin")
with open("finetune_llamindex_v3_docstore_metadata.pkl", "wb") as f:
    pickle.dump(index_to_docstore_id, f)



loaded_index = faiss_database.read_index("finetune_llamindex_v3_faiss_index.bin")

with open("finetune_llamindex_v3_docstore_metadata.pkl", "rb") as f:
    loaded_metadata = pickle.load(f)
    
if verbose:
    
    print("-----------------------------vector index made!---------------------------------")
    
if verbose:  
    
    print("------------------Evaluating metrics-----------------------")
    
metrics = {}
metrics['precision'] = {}
metrics['recall'] = {}
metrics['relevancy'] = {}

N = 5000
K_list = [3, 5, 10, 100]
for K in K_list:
    metrics['precision'][K] = []
    metrics['recall'][K] = []
    metrics['relevancy'][K] = []

np.random.seed(0)
indices = np.random.choice(len(test_df), N)
for i in tqdm(indices):

    query_text = test_df['query'][i]
    query_embedding = embedding_function(query_text).numpy().astype('float32').reshape(1, -1)
    distances, indices = loaded_index.search(query_embedding, K_list[-1])
    retrieved_passages = [loaded_metadata[j] for j in indices[0]]

    for K in K_list:
        metrics['precision'][K].append(contextual_precision(retrieved_passages[:K], test_df['passages'][i]['passage_text']))
        metrics['recall'][K].append(contextual_recall(retrieved_passages[:K], test_df['passages'][i]['passage_text']))
        metrics['relevancy'][K].append(contextual_relevancy(retrieved_passages[:K], test_df['passages'][i]['passage_text']))

  
import pickle

with open(f'metrics_5000_finetuned_llama-index_v3.pkl', 'wb') as f:
    pickle.dump(metrics, f)
    
with open('metrics_5000_finetuned_llama-index_v3.pkl', 'rb') as f:
    
    results = pickle.load(f)

fin_results = {}

for key in results.keys():
    
    fin_results[key] = {}
    
    for k in results[key].keys():
        
        fin_results[key][k] = sum(results[key][k])/len(results[key][k])
        
print(fin_results)
fin_results = pd.DataFrame(fin_results)

if verbose:
    
    print("-----------------------------Metrics evaluated!---------------------------------")
