import tensorflow as tf

class PatchesSpectra_v2(tf.keras.layers.Layer):
    def __init__(self, num_patches, embedding_dim):
        super(PatchesSpectra_v2, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=embedding_dim)
    
    def call(self, inputs):
        
        
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=-1)


        batch_size = tf.shape(inputs)[0]
        input_length = tf.shape(inputs)[1]
        patch_dim = input_length // self.num_patches
        
        
        patches = []
        for i in range(self.num_patches):
            patch = tf.reshape(
                inputs[:, i * patch_dim: (i + 1) * patch_dim, :],
                shape=(batch_size, patch_dim, 1, 1)
            )
            patches.append(patch)
        
        X_out = tf.concat(patches, axis=3)
        
        temp = tf.reshape(X_out, shape=(batch_size, self.num_patches, patch_dim))
        
        return self.projection(temp)

class AddCLS_L_Positional_v2(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_patches):
        super(AddCLS_L_Positional_v2, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches


        self.cls_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token"
        )

        self.positional_encodings = self.add_weight(
            shape=(1, num_patches + 1, embedding_dim),
            initializer="zeros",
            trainable=True,
            name="positional_encodings"
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])

        patch_embeddings = tf.concat([cls_tokens, inputs], axis=1)

       
        patch_embeddings = patch_embeddings + self.positional_encodings

        return patch_embeddings


class Transf_Block_v2(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, mlp_dim, rate = 0.2):

        super(Transf_Block_v2, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim=embedding_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(units=mlp_dim, activation='gelu'),
            tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(units=embedding_dim),
            tf.keras.layers.Dropout(rate),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-4)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-4)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training = True):
        att_out = self.att(self.layernorm1(inputs), self.layernorm1(inputs)) / tf.sqrt(tf.cast(inputs.shape[-1], tf.float32))

        out1 =  self.dropout(att_out) + inputs
    
        ffn_out = self.ffn(out1, training = training)

        return self.layernorm2(ffn_out) + out1


class AddCLS_L_Positional_v3(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_patches, num_tokens = 1):
        super(AddCLS_L_Positional_v3, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.num_tokens = num_tokens


        self.cls_tokens = self.add_weight(
            shape=(self.num_tokens, 1, self.embedding_dim),
            initializer="zeros",
            trainable=True,
            name="cls_tokens"
            )


        self.positional_encodings = self.add_weight(
            shape=(1, self.num_patches + self.num_tokens, self.embedding_dim),
            initializer="zeros",
            trainable=True,
            name="positional_encodings"
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        tokens = tf.tile(tf.transpose(self.cls_tokens, perm = [1, 0, 2]), [batch_size, 1, 1])
        
        patch_embeddings = tf.concat([tokens, inputs], axis=1)
        
        patch_embeddings = patch_embeddings + self.positional_encodings

        return patch_embeddings

