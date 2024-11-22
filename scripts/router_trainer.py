from torch import nn
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import  DataLoader
from tqdm import tqdm

verbose = True

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

if verbose:
    
    print("-------------------------Creating Dataset--------------------------------")
    
    
train_df_1 = pd.read_parquet('train-00005-of-00007.parquet')

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

ONE_HOT_MAP = {
    'DESCRIPTION': 0,
    'NUMERIC': 1,
    'ENTITY': 2,
    'LOCATION': 3,
    'PERSON': 4
    }


router_training_data = []

for _ , row in train_df_1.iterrows():
    
    label = torch.zeros((5,))
    label[ONE_HOT_MAP[row['query_type']]] = 1
    
    embedding = model.encode(row['query'])
    embedding /= np.linalg.norm(embedding)
    
    router_training_data.append([embedding, label])

# Custom Dataset
class RouterDataset:
    def __init__(self, training_data):
        """
        Args:
            training_data (list): List of [embedding, label] pairs.
        """
        self.data = training_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding, label = self.data[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Initialize the dataset
router_dataset = RouterDataset(router_training_data)

# Create DataLoader
router_dataloader = DataLoader(
    router_dataset,
    batch_size=64,  # Adjust batch size as needed
    shuffle=True,   # Shuffle for training
)

if verbose:
    
    print("-----------------------------Dataset created!-------------------------------\n\n")
    
    print("--------------------------Router training start!----------------------------\n")


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
    
router_model = Router().to(device)

  
max_epochs = 70
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(router_model.parameters(), lr = 1e-4)

for epoch in tqdm(range(max_epochs)):
    
    # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (embeddings, labels) in enumerate(router_dataloader):
        
        embeddings = embeddings.to(device)  # Move embeddings to GPU/CPU
        labels = labels.to(device)          # Move labels to GPU/CPU
        
        # Forward pass
        outputs = router_model(embeddings)  # Output shape: (batch_size, 5)
        target_labels = torch.argmax(labels, dim=1)  # Convert one-hot to class indices
        
        loss = criterion(outputs, target_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item()

    # Print epoch results
    avg_loss = running_loss / len(router_dataloader)
    print(f"Epoch [{epoch+1}/{max_epochs}], Loss: {avg_loss:.4f}")


if verbose:
    
    print("---------------------------------Router trained!--------------------------\n")
 

torch.save(router_model.state_dict(), 'chks/router/router_model_v2.pth')   
   