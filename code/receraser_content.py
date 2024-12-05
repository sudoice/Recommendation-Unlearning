import pickle
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings():
    # Load user and item embeddings from .pk files
    with open('user_common_embeddings.pk', 'rb') as f:
        user_common_embeddings = pickle.load(f)
    with open('item_common_embeddings.pk', 'rb') as f:
        item_common_embeddings = pickle.load(f)
    return user_common_embeddings, item_common_embeddings

def get_top_recommendations(user_id, user_common_embeddings, item_common_embeddings, top_n=10):
    # Check if user ID exists in the embeddings
    if user_id not in user_common_embeddings:
        raise ValueError("User ID {} not found in user embeddings.".format(user_id))


    # Get the user's embedding
    user_embedding = torch.tensor(user_common_embeddings[user_id]).unsqueeze(0)

    # Calculate similarity between the user embedding and each item embedding
    similarities = {}
    for item_id, item_embedding in item_common_embeddings.items():
        item_embedding_tensor = torch.tensor(item_embedding).unsqueeze(0)
        
        # Cosine similarity
        similarity = cosine_similarity(user_embedding, item_embedding_tensor)
        similarities[item_id] = similarity.item()  # Convert to scalar for sorting

    # Sort by similarity and get the top N items
    top_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Return top item IDs and their scores
    return [(item_id, score) for item_id, score in top_items]

# Load embeddings
user_common_embeddings, item_common_embeddings = load_embeddings()

# Specify the user ID and get recommendations
user_id = 1  # Example user ID
top_n = 10   # Number of recommendations

# Get top movie recommendations for the user

top_recommendations = get_top_recommendations(user_id, user_common_embeddings, item_common_embeddings, top_n)

# Display results
print("Top 10 Movie Recommendations:")
for item_id, score in top_recommendations:
  print("Movie ID: {}, Similarity Score: {}".format(item_id, score))
  
'''import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Content-Based Model Architecture
class ContentBasedModel(nn.Module):
    def _init_(self, embedding_dim):
        super(ContentBasedModel, self)._init_()
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, user_emb, item_emb):
        # Concatenate user and item embeddings
        x = torch.cat([user_emb, item_emb], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output relevance score
        return x

# Load shards from pickle files
def load_shards(part_type, part_num):
    shards = []
    try:
        # Load interaction data from partition files
        with open('../data/ml-1m/C_type-{}_num-{}.pk'.format(part_type, part_num), 'rb') as f:
            C = pickle.load(f)
        
        # For each cluster in C, create a list of (user_id, item_id) pairs
        for cluster_data in C:
            shard = []
            for user_id, items in cluster_data.items():
                for item_id in items:
                    shard.append((user_id, item_id))
            shards.append(shard)
    
    except Exception as e:
        print("Error loading shards: {}".format(e))
    
    return shards

def train_content_model_for_shards(shards, user_content_embeddings, item_content_embeddings, num_epochs=10):
    # To store models for each shard
    shard_models = []

    # Loop over each shard
    for shard_index, shard in enumerate(shards):
        print("Training content-based model for Shard {}...".format(shard_index))

        # Prepare data for this shard
        user_ids = [user_id for user_id, _ in shard]
        item_ids = [item_id for _, item_id in shard]

        # Extract the corresponding content embeddings for each user and item in the shard
        user_embs = torch.stack([torch.tensor(user_content_embeddings[user_id]) for user_id in user_ids])  # Shape: [num_users_in_shard, embedding_dim]
        # item_embs = torch.stack([torch.tensor(item_content_embeddings[item_id]) for item_id in item_ids])  # Shape: [num_items_in_shard, embedding_dim]
        # Define a fallback embedding for missing items (e.g., a zero vector of appropriate size)
        embedding_dim = user_embs.size(1)
        fallback_embedding = torch.zeros(embedding_dim)

        # Now modify the item_embs creation to handle missing item IDs gracefully
        item_embs = []
        for item_id in item_ids:
            # Check if item_id exists in item_content_embeddings
            if item_id in item_content_embeddings:
                item_embs.append(torch.tensor(item_content_embeddings[item_id]))
            else:
                # If the item is missing, append the fallback embedding
                item_embs.append(fallback_embedding)

        # Convert the list to a tensor
        item_embs = torch.stack(item_embs)
        
        # Define the target (assuming binary implicit feedback, 1 for interaction)
        targets = torch.ones(len(shard), 1)  # Shape: [num_pairs, 1]

        # Initialize model and optimizer
        model = ContentBasedModel(embedding_dim=embedding_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()  # Or use BCEWithLogitsLoss() for binary classification

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(user_embs, item_embs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print("Epoch {}/{} Loss: {}".format(epoch, num_epochs, loss.item()))

        # Append trained model to the shard models list
        shard_models.append(model)

        # Save model for this shard
        torch.save(model.state_dict(), 'shard_{}_content_model.pt'.format(shard_index))
        print("Model for Shard {} saved.".format(shard_index))

    return shard_models

# Load content embeddings from .pk files
with open('user_common_embeddings.pk', 'rb') as f:
    user_content_embeddings = pickle.load(f)  # Dictionary {user_id: embedding}
with open('item_common_embeddings.pk', 'rb') as f:
    item_content_embeddings = pickle.load(f)  # Dictionary {item_id: embedding}

# Load shards from saved files
part_type = 1  # Set to the partition type you're using
part_num = 10 # Set to the partition number you're using
shards = load_shards(part_type, part_num)

# Train content-based submodels for each shard
shard_models = train_content_model_for_shards(shards, user_content_embeddings, item_content_embeddings)'''