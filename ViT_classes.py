import tensorflow as tf

class Patches_spectra_27_11(tf.keras.layers.Layer):
    
    """
    Create the patches from the initial spectra. The patches are then projected
    to dimension given by embedding_dim
    """

    def __init__(self, num_patches, embedding_dim):
        super(Patches_spectra_27_11, self).__init__()
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.dense = tf.keras.layers.Dense(units = self.embedding_dim)

    def call(self, inputs, project = True):
        if len(inputs.shape) < 3:
            inputs = tf.expand_dims(inputs, axis = -1)

        patch_dims = inputs.shape[1] // self.num_patches
        batch_patch = []


        for i in range(self.num_patches):
            patch = inputs[:,i * patch_dims: (i+1) * patch_dims, :]
            #print(patch.shape)
            batch_patch.append(patch)
        
        temp = tf.transpose(tf.concat(batch_patch, axis = -1), perm = [0, 2, 1])

        if project:
            return self.dense(temp)
        else: 
            return temp

class AddCLS_L_Positional_v2(tf.keras.layers.Layer):
    
    """
    Add positional encoding as learnable parameters the each patch. An additional 
    cls patch is created. This with store the info and is the only token used by the 
    MLP head
    """

    def __init__(self, embedding_dim, num_patches):
        super(AddCLS_L_Positional_v2, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches


        self.cls_token = self.add_weight(
            shape=(1, 1, embedding_dim), # <------ ADD THE SAME ENCODING THE ALL THE PATCHES?!
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
    
    """
    Usual transformer block introduced by Ashish Vaswani et al., in 'Attention is all you need'.
    
    """

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
    """
    This version is meant to work with multiple cls tokens
    """
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



class ViT(tf.keras.models.Model):
    
    """
    Enseble of the blocks defined above.
    """


    def __init__(self, num_outputs, sequence_size, patch_size, embedding_dim, num_heads, mlp_dim, num_layers, training = True, rate = 0.1, pool_3x = False):
        super(ViT, self).__init__()
        self.pool_3 = tf.keras.layers.MaxPooling1D(pool_size = 3)
        self.pool_3x = pool_3x

        self.num_patches = sequence_size // patch_size

        self.generate_patches = PatchesSpectra_v2(num_patches=self.num_patches, embedding_dim = embedding_dim)

        self.CLSandPositional = AddCLS_L_Positional_v2(num_patches = self.num_patches, embedding_dim = embedding_dim)

        self.transf_block = [
            Transf_Block_v2(embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim = mlp_dim, rate = rate) for _ in range(num_layers)
            ]

        self.reg_head = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon=1e-4),
            tf.keras.layers.Dense(units=64, activation='gelu'),
            tf.keras.layers.Dropout(rate = 0.1),
            tf.keras.layers.Dense(units = num_outputs)
        ])
    
    def call(self, inputs, training = True):
        if self.pool_3x:

            
            if len(inputs.shape) == 2:
                inputs = tf.expand_dims(inputs, axis=-1)


            inputs = self.pool_3(inputs)

        x = self.generate_patches(inputs)

        x = self.CLSandPositional(x)

        for block in self.transf_block:
            x = block(x, training = training)
        
        cls_tokens = x[:, 0, :]
        output = self.reg_head(cls_tokens)

        
        return output
    

