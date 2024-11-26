import json
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
from llama_index.embeddings.adapter.utils import TwoLayerNN
from metrics import contextual_precision, contextual_recall, contextual_relevancy
from tqdm import tqdm
import pickle
from sentence_transformers import CrossEncoder
from torch import nn
import torch

verbose = True

# Set to True if reranker is used after retrieving documents 
set_rerank = True

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


ONE_HOT_MAP = {
    'DESCRIPTION': 0,
    'NUMERIC': 1,
    'ENTITY': 2,
    'LOCATION': 3,
    'PERSON': 4
    }

if verbose:
    
    print("-------------------------Initializing the router---------------------------")

# Use hidden dimension = 256, for v2 and 128 for v1
class Router(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=5):
        super(Router, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to hidden layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Hidden layer to output

    def forward(self, x):
        # Forward pass
        x = nn.functional.relu(self.fc1(x))  # Apply ReLU activation to hidden layer
        x = self.fc2(x)
        x = nn.functional.softmax(x) # Output layer
        return x
    
router = Router()

# Load router
state_dict = torch.load('chks/router/router_model_v3.pth', map_location=device)
router.load_state_dict(state_dict)

if verbose:
    
    print('----------------------------Router loaded-----------------------------')
    
# Load the base SentenceTransformer model
base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device = device)

test_df = pd.read_parquet('test-00000-of-00001.parquet')

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid())


def rerank(query_text, retrieved_documents):
    
    query_answer_pairs = [(query_text, passage) for passage in retrieved_documents]
    scores = reranker_model.predict(query_answer_pairs)
    sorted_passages = [p for p, s in sorted(zip(retrieved_documents, scores), key=lambda x: x[1], reverse=True)]
    
    return sorted_passages

index = faiss.IndexFlatIP(384)

def embedding_function(text):

    embeddings = base_model.encode(text)
    embeddings = adapter_model.forward(torch.tensor(embeddings, dtype = torch.float32, requires_grad=False)).detach()
    return embeddings / np.linalg.norm(embeddings)

vector_store = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


# Store experts
adapter_models = []

# Store faiss databases corresponding to expert
faiss_indexes = []
faiss_metadatas = []

for query_type in ONE_HOT_MAP.keys():
    
    if verbose:
        
        print(f"-------------------- Load {query_type} finetuned model ------------------------")
                
    model_path = f"chks/llama_index/all-MiniLM-L6-v2-finetuned-TwoLayerNN-{query_type}"
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
    
    adapter_models.append(adapter_model)

    loaded_index = faiss.read_index(f"finetune_llamaindex_faiss_index_{query_type}.bin")

    with open(f"finetune_llamaindex_docstore_metadata_{query_type}.pkl", "rb") as f:
        loaded_metadata = pickle.load(f)

    faiss_indexes.append(loaded_index)
    faiss_metadatas.append(loaded_metadata)

    
def adapter_specific_embedding_function(text, adapter_model):

    embeddings = base_model.encode(text)
    embeddings = adapter_model.forward(torch.tensor(embeddings, dtype = torch.float32, requires_grad=False)).detach()
    return embeddings / np.linalg.norm(embeddings)

    
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
    
    # Get query type from router
    embedding = base_model.encode(query_text)
    embedding /= np.linalg.norm(embedding)
    output = router.forward(torch.tensor(embedding))
    pred_label = torch.argmax(output)
    
    # Use corresponding expert for retrieval
    query_embedding = adapter_specific_embedding_function(query_text, adapter_models[pred_label]).numpy().astype('float32').reshape(1, -1)
    distances, indices = faiss_indexes[pred_label].search(query_embedding, K_list[-1])
    retrieved_passages = [faiss_metadatas[pred_label][j] for j in indices[0]] 
      
    if set_rerank:
        retrieved_passages = rerank(query_text, retrieved_passages)

    for K in K_list:
        metrics['precision'][K].append(contextual_precision(retrieved_passages[:K], test_df['passages'][i]['passage_text']))
        metrics['recall'][K].append(contextual_recall(retrieved_passages[:K], test_df['passages'][i]['passage_text']))
        metrics['relevancy'][K].append(contextual_relevancy(retrieved_passages[:K], test_df['passages'][i]['passage_text']))


if set_rerank:
    
    with open(f'metrics_5000_mixture_of_experts_reranked.pkl', 'wb') as f:
        pickle.dump(metrics, f)
        
    with open(f'metrics_5000_mixture_of_experts_reranked.pkl', 'rb') as f:
        
        results = pickle.load(f)

else:
    
    with open(f'metrics_5000_mixture_of_experts.pkl', 'wb') as f:
        pickle.dump(metrics, f)
        
    with open(f'metrics_5000_mixture_of_experts.pkl', 'rb') as f:
        
        results = pickle.load(f) 

fin_results = {}

for key in results.keys():
    
    fin_results[key] = {}
    
    for k in results[key].keys():
        
        fin_results[key][k] = sum(results[key][k])/len(results[key][k])
        
print(fin_results)

if verbose:
    
    print(f"-----------------------------Metrics evaluated !---------------------------------")