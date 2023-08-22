import torch
import math
import torch.nn as nn

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute pos embeddings
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        for pos in range(max_len):
            # For each dim of each position
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
            
        # include the batch size
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        return self.pe

class BERTEmbedding(nn.Module):
    """
    BERT Embedding consists of the following sub embeddings
        1. TokenEmbedding: Normal embedding matrix
        2. PositionalEmbedding: Adding positional information using sin, cos
        3. SegmentEmbedding: Adding sentence segment info
        Sum of all gives BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len=64, dropout=0.1):

        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = nn.Embedding(3, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
