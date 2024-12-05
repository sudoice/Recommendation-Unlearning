import tensorflow as tf
import pickle
import numpy as np

def load_embeddings():
    # Load user and item embeddings from .pk files
    with open('user_common_embeddings.pk', 'rb') as f:
        user_common_embeddings = pickle.load(f)
    with open('item_common_embeddings.pk', 'rb') as f:
        item_common_embeddings = pickle.load(f)
    return user_common_embeddings, item_common_embeddings

def init_weights(n_users, n_items, emb_dim, attention_size, num_local):
    all_weights = dict()
    initializer = tf.random_normal_initializer(stddev=0.01)
    user_common_embeddings, item_common_embeddings = load_embeddings()
    
    # Initialize 'user_embedding' with random values, shape (n_users, num_local, emb_dim)
    user_embedding_values = np.zeros((n_users, num_local, emb_dim), dtype=np.float32)

    # Populate user_embedding with user_common_embeddings for each user
    for user_id in range(n_users):
        if user_id in user_common_embeddings:
            embedding = user_common_embeddings[user_id]
            # Ensure the embedding has the correct shape (emb_dim,)
            embedding_resized = np.resize(embedding, emb_dim)
            embedding_resized_tensor = tf.convert_to_tensor(embedding_resized, dtype=tf.float32)
            embedding_resized_tensor = tf.reshape(embedding_resized_tensor, [1, emb_dim])  # Reshape to rank 2 tensor
            user_embedding_values[user_id] = tf.tile(embedding_resized_tensor, [num_local, 1])  # Tile the tensor along the first dimension
        else:
            # If the user ID does not have a corresponding embedding, initialize with random values
            random_embedding = initializer([emb_dim])  # Random embedding of correct size
            random_embedding_tensor = tf.convert_to_tensor(random_embedding, dtype=tf.float32)
            random_embedding_tensor = tf.reshape(random_embedding_tensor, [1, emb_dim])  # Reshape to rank 2 tensor
            user_embedding_values[user_id] = tf.tile(random_embedding_tensor, [num_local, 1])  # Tile the tensor along the first dimension

    all_weights['user_embedding'] = tf.Variable(user_embedding_values, name='user_embedding')

    # Initialize 'item_embedding' with random values, shape (n_items, num_local, emb_dim)
    item_embedding_values = np.zeros((n_items, num_local, emb_dim), dtype=np.float32)

    # Populate item_embedding with item_common_embeddings for each item
    for item_id in range(n_items):
        if item_id in item_common_embeddings:
            embedding = item_common_embeddings[item_id]
            # Ensure the embedding has the correct shape (emb_dim,)
            embedding_resized = np.resize(embedding, emb_dim)
            embedding_resized_tensor = tf.convert_to_tensor(embedding_resized, dtype=tf.float32)
            embedding_resized_tensor = tf.reshape(embedding_resized_tensor, [1, emb_dim])  # Reshape to rank 2 tensor
            item_embedding_values[item_id] = tf.tile(embedding_resized_tensor, [num_local, 1])  # Tile the tensor along the first dimension
        else:
            # If the item ID does not have a corresponding embedding, initialize with random values
            random_embedding = initializer([emb_dim])  # Random embedding of correct size
            random_embedding_tensor = tf.convert_to_tensor(random_embedding, dtype=tf.float32)
            random_embedding_tensor = tf.reshape(random_embedding_tensor, [1, emb_dim])  # Reshape to rank 2 tensor
            item_embedding_values[item_id] = tf.tile(random_embedding_tensor, [num_local, 1])  # Tile the tensor along the first dimension

    all_weights['item_embedding'] = tf.Variable(item_embedding_values, name='item_embedding')

    # User attention
    all_weights['WA'] = tf.Variable(
        tf.random.truncated_normal(shape=[emb_dim, attention_size], mean=0.0, stddev=tf.sqrt(
            tf.divide(2.0, attention_size + emb_dim))), dtype=tf.float32, name='WA')
    all_weights['BA'] = tf.Variable(tf.constant(0.00, shape=[attention_size]), name="BA")
    all_weights['HA'] = tf.Variable(tf.constant(0.01, shape=[attention_size, 1]), name="HA")

    # Item attention
    all_weights['WB'] = tf.Variable(
        tf.random.truncated_normal(shape=[emb_dim, attention_size], mean=0.0, stddev=tf.sqrt(
            tf.divide(2.0, attention_size + emb_dim))), dtype=tf.float32, name='WB')
    all_weights['BB'] = tf.Variable(tf.constant(0.00, shape=[attention_size]), name="BB")
    all_weights['HB'] = tf.Variable(tf.constant(0.01, shape=[attention_size, 1]), name="HB")

    # Transformation weights
    all_weights['trans_W'] = tf.Variable(initializer([num_local, emb_dim, emb_dim]), name='trans_W')
    all_weights['trans_B'] = tf.Variable(initializer([num_local, emb_dim]), name='trans_B')

    return all_weights





'''import tensorflow as tf

def load_embeddings():
    # Load user and item embeddings from .pk files
    with open('user_common_embeddings.pk', 'rb') as f:
        user_common_embeddings = pickle.load(f)
    with open('item_common_embeddings.pk', 'rb') as f:
        item_common_embeddings = pickle.load(f)
    return user_common_embeddings, item_common_embeddings

def init_weights(n_users, n_items, emb_dim, attention_size, num_local):
    all_weights = dict()
    initializer = tf.random_normal_initializer(stddev=0.01)

    all_weights['user_embedding'] = tf.Variable(initializer([n_users, num_local, emb_dim]), name='user_embedding')
    all_weights['item_embedding'] = tf.Variable(initializer([n_items, num_local, emb_dim]), name='item_embedding')

    # User attention
    all_weights['WA'] = tf.Variable(
        tf.random.truncated_normal(shape=[emb_dim, attention_size], mean=0.0, stddev=tf.sqrt(
            tf.divide(2.0, attention_size + emb_dim))), dtype=tf.float32, name='WA')
    all_weights['BA'] = tf.Variable(tf.constant(0.00, shape=[attention_size]), name="BA")
    all_weights['HA'] = tf.Variable(tf.constant(0.01, shape=[attention_size, 1]), name="HA")

    # Item attention
    all_weights['WB'] = tf.Variable(
        tf.random.truncated_normal(shape=[emb_dim, attention_size], mean=0.0, stddev=tf.sqrt(
            tf.divide(2.0, attention_size + emb_dim))), dtype=tf.float32, name='WB')
    all_weights['BB'] = tf.Variable(tf.constant(0.00, shape=[attention_size]), name="BB")
    all_weights['HB'] = tf.Variable(tf.constant(0.01, shape=[attention_size, 1]), name="HB")

    # Transformation weights
    all_weights['trans_W'] = tf.Variable(initializer([num_local, emb_dim, emb_dim]), name='trans_W')
    all_weights['trans_B'] = tf.Variable(initializer([num_local, emb_dim]), name='trans_B')

   

    return all_weights'''

