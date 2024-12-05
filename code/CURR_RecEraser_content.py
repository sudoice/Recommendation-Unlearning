import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define Dataset
class EmbeddingDataset(Dataset):
    def __init__(self, user_embeddings, item_embeddings):
        self.user_embeddings = torch.tensor(user_embeddings, dtype=torch.float32)
        self.item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.user_embeddings)

    def __getitem__(self, idx):
        return self.user_embeddings[idx], self.item_embeddings[idx]

# Define the Recommendation Model
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
        modified_user_embedding = self.user_fc(user_embedding)
        modified_item_embedding = self.item_fc(item_embedding)
        return modified_user_embedding, modified_item_embedding

# Train the Model
def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for user_embedding, item_embedding in dataloader:
            optimizer.zero_grad()

            # Forward pass
            modified_user, modified_item = model(user_embedding, item_embedding)
            
            # Compute loss (here we aim for similarity)
            loss = criterion(modified_user, user_embedding) + criterion(modified_item, item_embedding)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Example usage
if __name__ == "__main__":
    # Example data (replace with your embeddings)
    num_users = 1000
    num_items = 1000
    embedding_dim = 64

    user_embeddings = torch.rand(num_users, embedding_dim)
    item_embeddings = torch.rand(num_items, embedding_dim)

    # Dataset and DataLoader
    dataset = EmbeddingDataset(user_embeddings, item_embeddings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, Optimizer, and Loss
    model = ContentBasedModel(embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Loss for similarity

    # Train the model
    train_model(model, dataloader, optimizer, criterion, epochs=10)

    # Save or integrate modified embeddings
    model.eval()
    modified_user_embeddings = []
    modified_item_embeddings = []
    with torch.no_grad():
        for user_embedding, item_embedding in dataloader:
            modified_user, modified_item = model(user_embedding, item_embedding)
            modified_user_embeddings.append(modified_user)
            modified_item_embeddings.append(modified_item)

    # Concatenate results
    modified_user_embeddings = torch.cat(modified_user_embeddings)
    modified_item_embeddings = torch.cat(modified_item_embeddings)

    print("Modified user embeddings shape:", modified_user_embeddings.shape)
    print("Modified item embeddings shape:", modified_item_embeddings.shape)
