from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
import pandas as pd
from llama_index.embeddings.adapter.utils import TwoLayerNN
from llama_index.core.embeddings import resolve_embed_model


train_df_1 = pd.read_parquet('train-00000-of-00007.parquet')
train_df_2 = pd.read_parquet('train-00001-of-00007.parquet')
train_df_3 = pd.read_parquet('train-00002-of-00007.parquet')
train_df_4 = pd.read_parquet('train-00003-of-00007.parquet')
train_df_5 = pd.read_parquet('train-00004-of-00007.parquet')
train_df_6 = pd.read_parquet('train-00005-of-00007.parquet')
train_df_7 = pd.read_parquet('train-00006-of-00007.parquet')

NUM_EXAMPLES = {
    'DESCRIPTION': 150000,
    'NUMERIC': 105000,
    'ENTITY': 63243,
    'LOCATION': 46879,
    'PERSON': 42800
    }

for query_type in NUM_EXAMPLES.keys():
    
    train_data = pd.concat([train_df_1, train_df_2, train_df_3, train_df_4, train_df_5, train_df_6], axis=0)
    filtered_df = train_data[train_data['query_type'] == query_type]
    filtered_df = filtered_df.sample(NUM_EXAMPLES[query_type])

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

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    base_embed_model = resolve_embed_model(f"local:{model_name}")

    adapter_model = TwoLayerNN(
        in_features=384,
        hidden_features=512,
        out_features=384,
        bias=True,
        add_residual=True
    )

    finetune_engine = EmbeddingAdapterFinetuneEngine(
        train_dataset,
        base_embed_model,
        model_output_path=f"chks/llama_index/all-MiniLM-L6-v2-finetuned-TwoLayerNN-{query_type}", # Replace with your path
        adapter_model=adapter_model,
        epochs=5,
        verbose=False,
        device="cuda",
        batch_size = 64
    )

    print('-----------------Fine-Tuning start-----------------')
    finetune_engine.finetune()

