from torch import nn
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

verbose = True

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

if verbose:
    
    print("--------------------Loading models-------------------")
    
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

ONE_HOT_MAP = {
    'DESCRIPTION': 0,
    'NUMERIC': 1,
    'ENTITY': 2,
    'LOCATION': 3,
    'PERSON': 4
    }

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

state_dict = torch.load('chks/router/router_model_v2.pth', map_location=device)
router.load_state_dict(state_dict)

if verbose:
    
    print("--------------------Models Loaded!-------------------")
    
if verbose:
    
    print("-------------------------Evaluating router--------------------------------")

test_df = pd.read_parquet('test-00000-of-00001.parquet')
   
correct = 0

for _ , row in test_df.iterrows():
    
    
    true_label = ONE_HOT_MAP[row['query_type']]

    embedding = model.encode(row['query'])
    embedding /= np.linalg.norm(embedding)
    
    output = router.forward(torch.tensor(embedding))
    
    pred_label = torch.argmax(output)
    
    if true_label == pred_label:
        
        correct +=1
        

print(f"Accuracy of router: {correct/len(test_df)}")    

    
