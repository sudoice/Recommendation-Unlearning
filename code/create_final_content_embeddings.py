import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os

# Define Dataset
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, user_embeddings, item_embeddings):
        self.user_embeddings = torch.FloatTensor(list(user_embeddings.values()))  # Convert dict values to list
        self.item_embeddings = torch.FloatTensor(list(item_embeddings.values()))  # Convert dict values to list
    
    def __len__(self):
        # Here we return the max of the lengths of user and item embeddings
        return max(len(self.user_embeddings), len(self.item_embeddings))
    
    def __getitem__(self, idx):
        # Ensure we only return available embeddings
        user_embedding = self.user_embeddings[idx] if idx < len(self.user_embeddings) else None
        item_embedding = self.item_embeddings[idx] if idx < len(self.item_embeddings) else None
        
        # Ensure neither is None
        if user_embedding is None or item_embedding is None:
            raise ValueError("Embedding at index {} is missing for user or item.".format(idx))
        
        return user_embedding, item_embedding


# Define the Recommendation Model for a single shard
class ContentBasedModel(nn.Module):
    def __init__(self, embedding_dim):
        super(ContentBasedModel, self).__init__()
        # Define layers to modify embeddings
        self.user_fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.item_fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, user_embedding, item_embedding):
        if user_embedding is not None:
            modified_user_embedding = self.user_fc(user_embedding)
        else:
            modified_user_embedding = None
            
        if item_embedding is not None:
            modified_item_embedding = self.item_fc(item_embedding)
        else:
            modified_item_embedding = None
            
        return modified_user_embedding, modified_item_embedding

# Train the Model for a shard
def train_shard_model(model, dataloader, optimizer, criterion, epochs, shard_id):
    model.train()
    for epoch in xrange(epochs):  # Use xrange in Python 2 for better memory efficiency
        epoch_loss = 0
        for user_embedding, item_embedding in dataloader:
            optimizer.zero_grad()

            # Forward pass
            modified_user, modified_item = model(user_embedding, item_embedding)
            
            # Compute loss (here we aim for similarity)
            loss = 0
            if modified_user is not None:
                loss += criterion(modified_user, user_embedding)
            if modified_item is not None:
                loss += criterion(modified_item, item_embedding)
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print "Shard {} - Epoch {}/{}, Loss: {:.4f}".format(shard_id, epoch+1, epochs, epoch_loss)  # Python 2 print statement

# Process each shard
def process_shard(shard_id, user_embeddings, item_embeddings, embedding_dim, output_dir, epochs=10, batch_size=32, lr=0.001):
    # Dataset and DataLoader for the shard
    dataset = EmbeddingDataset(user_embeddings, item_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Optimizer, and Loss
    model = ContentBasedModel(embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train the model for the shard
    train_shard_model(model, dataloader, optimizer, criterion, epochs, shard_id)

    # Save the resultant embeddings
    model.eval()
    modified_user_embeddings = []
    modified_item_embeddings = []
    with torch.no_grad():
        for user_embedding, item_embedding in dataloader:
            modified_user, modified_item = model(user_embedding, item_embedding)
            if modified_user is not None:
                modified_user_embeddings.append(modified_user)
            if modified_item is not None:
                modified_item_embeddings.append(modified_item)

    # Concatenate results
    modified_user_embeddings = torch.cat(modified_user_embeddings).numpy()
    modified_item_embeddings = torch.cat(modified_item_embeddings).numpy()

    # Save to .pk files
    user_path = os.path.join(output_dir, "user_content_embeddings_final_{}.pk".format(shard_id))
    item_path = os.path.join(output_dir, "item_content_embeddings_final_{}.pk".format(shard_id))
    with open(user_path, 'wb') as user_file:
        pickle.dump(modified_user_embeddings, user_file)
    with open(item_path, 'wb') as item_file:
        pickle.dump(modified_item_embeddings, item_file)
    print "Shard {} embeddings saved to {} and {}".format(shard_id, user_path, item_path)  # Python 2 print statement

# Main function to handle multiple shards
if __name__ == "__main__":
    # Example configuration
    num_shards = 5
    embedding_dim = 32
    output_dir = "shard_embeddings"

    # Manually check if the directory exists before creating it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each shard using pre-existing embeddings
    for shard_id in xrange(num_shards):  # Use xrange in Python 2
        # Load user and item embeddings from .pk files
        user_embeddings_path = "sharded_embeddings/user_embeddings_shard_{}.pk".format(shard_id)
        item_embeddings_path = "sharded_embeddings/item_embeddings_shard_{}.pk".format(shard_id)

        # Load embeddings
        with open(user_embeddings_path, 'rb') as user_file:
            user_embeddings = pickle.load(user_file)
        
        with open(item_embeddings_path, 'rb') as item_file:
            item_embeddings = pickle.load(item_file)
        
        # Process the shard
        process_shard(shard_id, user_embeddings, item_embeddings, embedding_dim, output_dir)
