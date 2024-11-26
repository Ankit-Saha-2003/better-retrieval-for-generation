from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
import pandas as pd
from llama_index.embeddings.adapter.utils import TwoLayerNN
from llama_index.core.embeddings import resolve_embed_model
import json
import torch

verbose = True

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
  
train_df_1 = pd.read_parquet('train-00000-of-00007.parquet')
train_df_2 = pd.read_parquet('train-00001-of-00007.parquet')
train_df_3 = pd.read_parquet('train-00002-of-00007.parquet')
train_df_4 = pd.read_parquet('train-00003-of-00007.parquet')
train_df_5 = pd.read_parquet('train-00004-of-00007.parquet')
train_df_6 = pd.read_parquet('train-00005-of-00007.parquet')
train_df_7 = pd.read_parquet('train-00006-of-00007.parquet')

NUM_EXAMPLES = {
    'DESCRIPTION': 100000,
    'NUMERIC': 105000,
    'ENTITY': 63243,
    'LOCATION': 46879,
    'PERSON': 42800
    }

train_data = pd.concat([train_df_1, train_df_2, train_df_3, train_df_4, train_df_5, train_df_6], axis=0)

for query_type in NUM_EXAMPLES.keys():
    
    if verbose:
        
        print('---------------------Loading dataset---------------------------')
    
    filtered_df = train_data[train_data['query_type'] == query_type]
    filtered_df = filtered_df.sample(NUM_EXAMPLES[query_type])
    
    # # Save dataframe
    # file_path = f"{query_type}-contextual_metrics.csv"
    # filtered_df.to_csv(file_path, index=False)
    
    train_queries = dict()
    corpus = dict()
    train_relevant_docs = dict()

    count = 0

    for index, row in filtered_df.iterrows():

        train_queries[f'{index}'] = row['query']

        for corpus_index, passage in enumerate(row['passages']['passage_text']):

            corpus[f'{index}.{corpus_index}'] = passage

        train_relevant_docs[f'{index}'] = [f'{index}.{i}' for i in range(len(row['passages']['passage_text']))]
        
        
    train_dataset = EmbeddingQAFinetuneDataset(
        queries = train_queries, corpus = corpus, relevant_docs = train_relevant_docs
    )
    if verbose:
        
        print('----------------------------Dataset loaded------------------------------------')
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    base_embed_model = resolve_embed_model(f"local:{model_name}")

    adapter_model = TwoLayerNN(
        in_features=384,
        hidden_features=768,
        out_features=384,
        bias=True,
        add_residual=True
    )
    
    ##  Uncomment if you want to continue from a checkpoint
    # model_path = f"chks/llama_index/all-MiniLM-L6-v2-finetuned-TwoLayerNN-{query_type}"
    # config_path = f"{model_path}/config.json"
    # model_weights_path = f"{model_path}/pytorch_model.bin"

    # # Load the config
    # with open(config_path, "r") as f:
    #     config = json.load(f)

    # # Reconstruct the TwoLayerNN adapter model
    # adapter_model = TwoLayerNN(
    #     in_features=config["in_features"],
    #     hidden_features=config["hidden_features"],
    #     out_features=config["out_features"],
    #     bias=config["bias"],
    #     activation_fn_str=config["activation_fn_str"],
    #     add_residual=config["add_residual"]
    # )

    ##  Load the adapter model's weights
    # adapter_model.load_state_dict(torch.load(model_weights_path,  map_location=torch.device(device)))
    
    finetune_engine = EmbeddingAdapterFinetuneEngine(
        train_dataset,
        base_embed_model,
        model_output_path=f"chks/llama_index/all-MiniLM-L6-v2-finetuned-TwoLayerNN-{query_type}", # Replace with your path
        adapter_model=adapter_model,
        epochs=15,
        verbose=False,
        device="cuda",
        batch_size = 64
    )
    if verbose:
    
        print('-----------------Fine-Tuning start-----------------')
        
    finetune_engine.finetune()

