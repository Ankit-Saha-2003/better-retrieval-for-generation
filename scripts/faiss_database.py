import json
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
from llama_index.embeddings.adapter.utils import TwoLayerNN
from llama_index.core.embeddings import resolve_embed_model
from tqdm import tqdm
import pickle

NUM_EXAMPLES = {
    'DESCRIPTION': 150000,
    'NUMERIC': 105000,
    'ENTITY': 63243,
    'LOCATION': 46879,
    'PERSON': 42800
    }

for query_type in NUM_EXAMPLES.keys():
    
    print(f'----------------------------------{query_type}----------------------------------')
    print()
    print("--------------------Load finetuned model------------------------")
    model_path = F"chks/llama_index/all-MiniLM-L6-v2-finetuned-TwoLayerNN-{query_type}"
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

    index = faiss.IndexFlatIP(384)
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
    
    
    print(f'-------------------Saving vector index {query_type}-----------------------------------')
    faiss.write_index(index, f"finetune_llamaindex_faiss_index_{query_type}.bin")
    with open(f"finetune_llamaindex_v3_docstore_metadata_{query_type}.pkl", "wb") as f:
        pickle.dump(index_to_docstore_id, f)