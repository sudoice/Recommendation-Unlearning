import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import pickle
import os

def create_content_based_item_embeddings():
    # Load movies.dat
    movies = pd.read_csv('../data/ml-1m/movies.dat', sep='::', names=['MovieID', 'Title', 'Genres'], engine='python')

    # Split genres and apply one-hot encoding
    movies['Genres'] = movies['Genres'].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(movies['Genres'])

    # Convert to PyTorch tensor
    genres_tensor = torch.tensor(genres_encoded, dtype=torch.float32)

    # Define an embedding layer for genres
    class GenreEmbedding(nn.Module):
        def __init__(self, input_dim, embedding_dim):
            super(GenreEmbedding, self).__init__()
            self.embedding = nn.Linear(input_dim, embedding_dim)
        
        def forward(self, x):
            return self.embedding(x)

    embedding_dim = 16
    genre_embedding_model = GenreEmbedding(input_dim=genres_encoded.shape[1], embedding_dim=embedding_dim)
    item_embeddings = genre_embedding_model(genres_tensor)

    # Create a dictionary for item embeddings with MovieID as the key
    item_embedding_dict = {movie_id: item_embeddings[i].detach().numpy() for i, movie_id in enumerate(movies['MovieID'].values)}
    
    return item_embedding_dict

def create_content_based_user_embeddings():
    # Load users.dat
    users = pd.read_csv('../data/ml-1m/users.dat', sep='::', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python')

    # Encode gender
    users['Gender'] = users['Gender'].map({'M': 1, 'F': 0})

    # Define age mapping to indices
    age_mapping = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    users['AgeIndex'] = users['Age'].map(age_mapping)

    # One-Hot Encode Occupation
    occupation_onehot = pd.get_dummies(users['Occupation'], prefix='Occupation')

    # Define embedding layers for age and occupation
    class UserEmbedding(nn.Module):
        def __init__(self, num_age_categories, num_occupation_categories, embedding_dim):
            super(UserEmbedding, self).__init__()
            self.age_embedding = nn.Embedding(num_age_categories, embedding_dim)
            self.occupation_embedding = nn.Embedding(num_occupation_categories, embedding_dim)
        
        def forward(self, age_indices, occupation_indices):
            age_emb = self.age_embedding(age_indices)
            occupation_emb = self.occupation_embedding(occupation_indices)
            return torch.cat([age_emb, occupation_emb], dim=1)

    num_age_categories = 7
    num_occupation_categories = users['Occupation'].nunique()
    embedding_dim = 16

    user_embedding_model = UserEmbedding(num_age_categories, num_occupation_categories, embedding_dim)

    # Convert categorical indices to tensors
    age_indices = torch.tensor(users['AgeIndex'].values, dtype=torch.long)
    occupation_indices = torch.tensor(users['Occupation'].values, dtype=torch.long)

    user_embeddings = user_embedding_model(age_indices, occupation_indices)

    # Add gender as a separate feature and concatenate it with age and occupation embeddings
    gender_tensor = torch.tensor(users['Gender'].values, dtype=torch.float32).unsqueeze(1)
    user_full_embeddings = torch.cat([gender_tensor, user_embeddings], dim=1)

    # Create a dictionary for user embeddings with UserID as the key
    user_embedding_dict = {user_id: user_full_embeddings[i].detach().numpy() for i, user_id in enumerate(users['UserID'].values)}
    
    return user_embedding_dict

class CommonSpaceMapper(nn.Module):
    def __init__(self, user_dim, movie_dim, common_dim):
        super(CommonSpaceMapper, self).__init__()
        self.user_mapper = nn.Linear(user_dim, common_dim)
        self.movie_mapper = nn.Linear(movie_dim, common_dim)
    
    def forward(self, user_emb, movie_emb):
        user_mapped = self.user_mapper(user_emb)
        movie_mapped = self.movie_mapper(movie_emb)
        return user_mapped, movie_mapped

def save_embeddings_for_shards(user_common_embeddings, item_common_embeddings, C, output_dir="sharded_embeddings"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for shard_idx in xrange(len(C)):  # Python 2 uses xrange instead of range
        user_ids_in_shard = list(C[shard_idx].keys())
        item_ids_in_shard = []
        for user_id in user_ids_in_shard:
            item_ids_in_shard.extend(C[shard_idx][user_id])
        
        shard_user_embeddings = {user_id: user_common_embeddings[user_id] for user_id in user_ids_in_shard}
        shard_item_embeddings = {item_id: item_common_embeddings[item_id] for item_id in item_ids_in_shard}

        # Use .format() for string formatting, which works in Python 2
        with open("{}/user_embeddings_shard_{}.pk".format(output_dir, shard_idx), 'wb') as f:
            pickle.dump(shard_user_embeddings, f)

        with open("{}/item_embeddings_shard_{}.pk".format(output_dir, shard_idx), 'wb') as f:
            pickle.dump(shard_item_embeddings, f)


# Create content embeddings for users and items
user_embeddings = create_content_based_user_embeddings()
item_embeddings = create_content_based_item_embeddings()

user_emb_tensor = torch.tensor([embedding for embedding in user_embeddings.values()])
item_emb_tensor = torch.tensor([embedding for embedding in item_embeddings.values()])

common_dim = 32
mapper = CommonSpaceMapper(user_emb_tensor.shape[1], item_emb_tensor.shape[1], common_dim)
user_common, item_common = mapper(user_emb_tensor, item_emb_tensor)

user_common_dict = {user_id: user_common[i].detach().numpy() for i, user_id in enumerate(user_embeddings.keys())}
item_common_dict = {item_id: item_common[i].detach().numpy() for i, item_id in enumerate(item_embeddings.keys())}

C, _, _ = data_partition_1(train_items, k=5, T=10)

save_embeddings_for_shards(user_common_dict, item_common_dict, C)
