
import torch
import torch.nn as nn

class PronunciationEncoder(nn.Module):
    ###
    # | One-hot Content Prediction | -> | cat | -> | Projection Layer | -> | + | -> | Transformer Layers | -> | Dropout(0.3) | -> | Pronunciation Sequence |
    # | Accent ID | -> | Accent Embedding | -> | cat |
    # | Positional Encoding | -> | + |
    ###

    def __init__(self, embedding_dim, projection_dim, accent_vocab_size, num_transformer_layers=4, num_heads=8, dropout=0.3):
        super(PronunciationEncoder, self).__init__()
        # 1. accent ID -> accent Embedding
        self.accent_embedding = nn.Embedding(accent_vocab_size, embedding_dim)

        # 2. projection Layer
        # self.content_embedding = nn.Embedding(content_vocab_size, embedding_dim)
        self.projection_layer = nn.Linear(embedding_dim, projection_dim)

        # 3. positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1, projection_dim))

        # 4. transformer layers
        self.transformer_layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=projection_dim, nhead=num_heads, dropout=dropout), num_layers=num_transformer_layers)            

        # 5. dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, content_preds, accent_ids):
        # accent ID -> accent Embedding
        accent_embed = self.accent_embedding(accent_ids)

        # concatenation
        combined = torch.cat((content_preds, accent_embed), dim=-1)

        # cat -> projection Layer
        projection = self.projection_layer(combined)
        
        # adding positional encoding
        combined = projection + self.positional_encoding
        
        # transformer layers
        transformer_out = self.transformer_layers(combined)        
        
        # dropout
        output = self.dropout(transformer_out)

        return output