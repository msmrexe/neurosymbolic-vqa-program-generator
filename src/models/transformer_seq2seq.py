import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding for Transformers.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerSeq2Seq(nn.Module):
    """
    A Transformer-based Seq2Seq model.
    """
    def __init__(
        self,
        vocab: dict,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super(TransformerSeq2Seq, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Vocab info
        self.q_vocab_size = len(vocab["question_token_to_idx"])
        self.p_vocab_size = len(vocab["program_token_to_idx"])
        self.pad_id = vocab["program_token_to_idx"]["<NULL>"]
        self.start_id = vocab["program_token_to_idx"]["<START>"]
        self.end_id = vocab["program_token_to_idx"]["<END>"]

        # Embeddings
        self.question_embedding = nn.Embedding(self.q_vocab_size, d_model)
        self.program_embedding = nn.Embedding(self.p_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        # Core Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Use batch_first=True
        )
        
        # Output generator
        self.generator = nn.Linear(d_model, self.p_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _create_pad_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Creates a boolean padding mask. True where padded."""
        return seq == self.pad_id

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates a square causal mask for the decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, questions: torch.Tensor, programs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for supervised training.

        Args:
            questions (torch.Tensor): Source sequences. Shape: [B, S_q]
            programs (torch.Tensor): Target sequences (shifted right). Shape: [B, S_p]

        Returns:
            torch.Tensor: Logits. Shape: [B, S_p, V_p]
        """
        device = questions.device
        
        # 1. Create masks
        src_pad_mask = self._create_pad_mask(questions)  # [B, S_q]
        tgt_pad_mask = self._create_pad_mask(programs)   # [B, S_p]
        
        tgt_seq_len = programs.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(device) # [S_p, S_p]
        
        # 2. Embed and add positional encoding
        src_emb = self.pos_encoder(self.question_embedding(questions) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.program_embedding(programs) * math.sqrt(self.d_model))
        
        # 3. Pass through Transformer
        # Transformer expects batch_first=True
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
        )
        
        # 4. Generate logits
        logits = self.generator(output)
        return logits

    def encode(self, questions: torch.Tensor):
        """Encodes the source sequence."""
        src_pad_mask = self._create_pad_mask(questions)
        src_emb = self.pos_encoder(self.question_embedding(questions) * math.sqrt(self.d_model))
        
        # Pass through encoder
        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_pad_mask
        )
        return memory, src_pad_mask

    def decode(self, tgt_seq: torch.Tensor, memory: torch.Tensor, memory_pad_mask: torch.Tensor):
        """Performs one decoding step."""
        device = tgt_seq.device
        tgt_seq_len = tgt_seq.size(1)
        
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(device)
        tgt_pad_mask = self._create_pad_mask(tgt_seq)

        tgt_emb = self.pos_encoder(self.program_embedding(tgt_seq) * math.sqrt(self.d_model))

        # Pass through decoder
        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask
        )
        
        # Return logits for the *last* token
        return self.generator(output[:, -1, :]) # [B, V_p]

    def forward_sample(self, questions: torch.Tensor, reinforce_sample: bool = False):
        """
        Auto-regressive decoding for inference or REINFORCE.

        Args:
            questions (torch.Tensor): Source sequences. Shape: [B, S_q]
            reinforce_sample (bool): Whether to sample or use argmax.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - output_symbols: Sampled program indices. Shape: [B, max_len]
            - output_logprobs: Log probabilities of sampled tokens. Shape: [B, max_len]
        """
        batch_size = questions.size(0)
        device = questions.device
        
        # 1. Encode source sequence once
        memory, src_pad_mask = self.encode(questions)
        
        # 2. Initialize target sequence with <START> token
        tgt_seq = torch.tensor(
            [[self.start_id]] * batch_size, dtype=torch.long, device=device
        ) # [B, 1]
        
        output_symbols = []
        output_logprobs = []

        for _ in range(self.max_seq_len):
            # 3. Decode one step
            logits = self.decode(tgt_seq, memory, src_pad_mask) # [B, V_p]
            
            # 4. Get log probabilities
            log_prob = F.log_softmax(logits, dim=1) # [B, V_p]
            
            # 5. Sample next token
            if reinforce_sample:
                dist = torch.distributions.Categorical(logits=logits)
                next_symbol = dist.sample() # [B]
                step_log_prob = dist.log_prob(next_symbol) # [B]
            else:
                next_symbol = torch.argmax(log_prob, dim=1) # [B]
                step_log_prob = log_prob.gather(1, next_symbol.unsqueeze(-1)).squeeze(-1) # [B]

            # 6. Append to lists
            output_symbols.append(next_symbol)
            output_logprobs.append(step_log_prob)
            
            # 7. Add new token to target sequence for next step
            tgt_seq = torch.cat(
                [tgt_seq, next_symbol.unsqueeze(1)], dim=1
            ) # [B, t+1]

        # 8. Stack all steps
        output_symbols = torch.stack(output_symbols, dim=1) # [B, max_len]
        output_logprobs = torch.stack(output_logprobs, dim=1) # [B, max_len]
        
        return output_symbols, output_logprobs
