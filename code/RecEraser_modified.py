import pickle
import numpy as np

class RecEraser_LightGCN(object):
    def __init__(self, data_config, alpha=0.5):
        # existing code
        self.alpha = alpha  # weight factor for combining content and collaborative embeddings

        # Load content embeddings (update paths as necessary)
        self.user_content_embeddings = self.load_content_embeddings('user_content_embeddings.pk')
        self.item_content_embeddings = self.load_content_embeddings('item_content_embeddings.pk')

    def load_content_embeddings(self, file_path):
        """Load content-based embeddings from a pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def combine_embeddings(self, u_collab, i_collab, user_ids, item_ids):
        """Combine collaborative and content-based embeddings with weighting by alpha."""
        u_content = tf.nn.embedding_lookup(self.user_content_embeddings, user_ids)
        i_content = tf.nn.embedding_lookup(self.item_content_embeddings, item_ids)

        # Weighted sum of collaborative and content embeddings
        u_combined = self.alpha * u_collab + (1 - self.alpha) * u_content
        i_combined = self.alpha * i_collab + (1 - self.alpha) * i_content

        return u_combined, i_combined

    def train_single_model(self, local_num):
        # existing collaborative embeddings
        ua_embeddings, ia_embeddings = self._create_lightgcn_embed_local(local_num)

        # Combining with content-based embeddings
        u_combined, i_combined = self.combine_embeddings(ua_embeddings, ia_embeddings, self.users, self.pos_items)

        # use combined embeddings for positive and negative items
        u_g_embeddings = tf.nn.embedding_lookup(u_combined, self.users)
        pos_i_g_embeddings = tf.nn.embedding_lookup(i_combined, self.pos_items)
        neg_i_g_embeddings = tf.nn.embedding_lookup(i_combined, self.neg_items)

        # regularization and loss calculation
        mf_loss = self.create_bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        regularizer = tf.nn.l2_loss(u_g_embeddings) + tf.nn.l2_loss(pos_i_g_embeddings) + tf.nn.l2_loss(neg_i_g_embeddings)
        regularizer = regularizer / self.batch_size
        emb_loss = self.decay * regularizer
        loss = mf_loss + emb_loss

        # optimization and batch ratings
        batch_ratings = tf.matmul(u_g_embeddings, pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        
        return opt, loss, mf_loss, emb_loss, batch_ratings

    def train_agg_model(self):
        # Similar to train_single_model but with aggregated content-based embeddings across shards
        user_agg_emb, item_agg_emb = self.aggregate_shard_embeddings()

        u_g_embeddings = tf.nn.embedding_lookup(user_agg_emb, self.users)
        pos_i_g_embeddings = tf.nn.embedding_lookup(item_agg_emb, self.pos_items)
        neg_i_g_embeddings = tf.nn.embedding_lookup(item_agg_emb, self.neg_items)

        mf_loss = self.create_bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        loss = mf_loss
        batch_ratings = tf.matmul(u_g_embeddings, pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        return opt, loss, mf_loss, batch_ratings

    def aggregate_shard_embeddings(self):
        """Aggregate collaborative and content embeddings for all shards."""
        user_agg_embeddings, item_agg_embeddings = [], []

        for i in range(self.num_local):
            u_g_emb, i_g_emb = self._create_lightgcn_embed_local(i)
            u_combined, i_combined = self.combine_embeddings(u_g_emb, i_g_emb, range(self.n_users), range(self.n_items))

            user_agg_embeddings.append(u_combined)
            item_agg_embeddings.append(i_combined)

        user_agg_emb = tf.reduce_mean(tf.stack(user_agg_embeddings), axis=0)
        item_agg_emb = tf.reduce_mean(tf.stack(item_agg_embeddings), axis=0)

        return user_agg_emb, item_agg_emb
