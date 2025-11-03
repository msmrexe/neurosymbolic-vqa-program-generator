import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .base_rnn import BaseRNN, RNN_CELL_TYPES


class Attention(nn.Module):
    """
    Implements a Bahdanau-style (additive) or general (dot-product) attention.
    The original implementation was a mix; this is a cleaned-up version.
    This implementation uses the "general" attention score:
    score(h_t, h_s) = h_t^T * W_a * h_s
    """

    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        # Linear layer to apply to the encoder output (h_s)
        self.attn_weights = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Linear layer to combine context vector and decoder output
        self.context_combine = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, decoder_output: torch.Tensor, encoder_outputs: torch.Tensor):
        """
        Calculates attention scores and applies them.

        Args:
            decoder_output (torch.Tensor): Decoder output tensor.
                Shape: [batch_size, 1, hidden_dim]
            encoder_outputs (torch.Tensor): All encoder hidden states.
                Shape: [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - The attention-combined output. Shape: [batch_size, 1, hidden_dim]
            - The attention weights. Shape: [batch_size, 1, seq_len]
        """
        # Apply linear layer to encoder outputs
        # encoder_outputs -> [B, S, H]
        # processed_encoder -> [B, S, H]
        processed_encoder_outputs = self.attn_weights(encoder_outputs)
        
        # Calculate attention scores
        # (B, 1, H) bmm (B, H, S) -> (B, 1, S)
        attn_scores = torch.bmm(decoder_output, processed_encoder_outputs.transpose(1, 2))
        
        # Apply softmax to get weights
        attn_weights = F.softmax(attn_scores, dim=2)
        
        # Calculate context vector
        # (B, 1, S) bmm (B, S, H) -> (B, 1, H)
        context_vector = torch.bmm(attn_weights, encoder_outputs)
        
        # Concatenate context vector and decoder output
        # (B, 1, H*2)
        combined = torch.cat((context_vector, decoder_output), dim=2)
        
        # Combine and apply tanh
        # (B, 1, H)
        output = torch.tanh(self.context_combine(combined))
        
        return output, attn_weights


class LstmEncoder(BaseRNN):
    """LSTM Encoder"""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        word_vec_dim: int,
        hidden_size: int,
        num_layers: int,
        input_dropout_prob: float,
        rnn_dropout_prob: float,
        bidirectional: bool = False,
        rnn_cell: RNN_CELL_TYPES = "lstm",
        variable_lengths: bool = False,
    ):
        super(LstmEncoder, self).__init__(
            vocab_size, max_seq_len, hidden_size,
            input_dropout_prob, rnn_dropout_prob,
            num_layers, rnn_cell
        )
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, word_vec_dim)
        
        self.rnn = self.rnn_cell(
            word_vec_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=self.rnn_dropout_prob if num_layers > 1 else 0,
        )

    def forward(self, input_seqs: torch.Tensor, input_lengths: torch.Tensor = None):
        """
        Forward pass for the encoder.

        Args:
            input_seqs (torch.Tensor): Input sequences. Shape: [B, S]
            input_lengths (torch.Tensor, optional): Lengths of sequences. Shape: [B]

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
            - Encoder outputs. Shape: [B, S, H * (2 if bidirectional)]
            - Encoder hidden state.
        """
        embedded = self.embedding(input_seqs)
        embedded = self.input_dropout(embedded)
        
        if self.variable_lengths and input_lengths is not None:
            # Pack padded sequence
            packed = pack_padded_sequence(
                embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            # Pass packed sequence through RNN
            packed_outputs, hidden = self.rnn(packed)
            # Unpack sequence
            outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            # Pass sequence through RNN
            outputs, hidden = self.rnn(embedded)
            
        return outputs, hidden


class LstmDecoder(BaseRNN):
    """LSTM Decoder with Attention"""

    def __init__(
        self,
        vocab: dict,
        word_vec_dim: int,
        hidden_size: int,
        num_layers: int,
        input_dropout_prob: float,
        rnn_dropout_prob: float,
        bidirectional_encoder: bool = False,
        rnn_cell: RNN_CELL_TYPES = "lstm",
        use_attention: bool = True,
    ):
        vocab_size = len(vocab["program_token_to_idx"])
        max_seq_len = 30  # Max program length
        
        super(LstmDecoder, self).__init__(
            vocab_size, max_seq_len, hidden_size,
            input_dropout_prob, rnn_dropout_prob,
            num_layers, rnn_cell
        )
        
        self.start_id = vocab["program_token_to_idx"]["<START>"]
        self.end_id = vocab["program_token_to_idx"]["<END>"]
        
        self.bidirectional_encoder = bidirectional_encoder
        self.encoder_hidden_dim = hidden_size * (2 if bidirectional_encoder else 1)
        self.use_attention = use_attention
        
        self.embedding = nn.Embedding(self.vocab_size, word_vec_dim)
        
        self.rnn = self.rnn_cell(
            word_vec_dim,
            self.encoder_hidden_dim, # Decoder hidden must match encoder hidden
            num_layers,
            batch_first=True,
            dropout=self.rnn_dropout_prob if num_layers > 1 else 0,
        )
        
        self.out_linear = nn.Linear(self.encoder_hidden_dim, self.vocab_size)
        
        if use_attention:
            self.attention = Attention(self.encoder_hidden_dim)
        else:
            self.attention = None

    def _format_encoder_hidden(self, encoder_hidden):
        """Helper to adjust bidirectional encoder hidden state for decoder."""
        if not self.bidirectional_encoder:
            return encoder_hidden

        # Handle LSTM (h, c)
        if isinstance(encoder_hidden, tuple):
            h, c = encoder_hidden
            # Concatenate forward and backward states
            # h shape: [num_layers * 2, B, H]
            num_layers, batch_size, hidden_size = h.size(1) // 2, h.size(1), h.size(2)
            h = h.view(self.num_layers, 2, batch_size, hidden_size) \
                 .transpose(1, 2).contiguous() \
                 .view(self.num_layers, batch_size, hidden_size * 2)
            c = c.view(self.num_layers, 2, batch_size, hidden_size) \
                 .transpose(1, 2).contiguous() \
                 .view(self.num_layers, batch_size, hidden_size * 2)
            return (h, c)
        else: # Handle GRU (h)
            # h shape: [num_layers * 2, B, H]
            num_layers, batch_size, hidden_size = h.size(1) // 2, h.size(1), h.size(2)
            h = h.view(self.num_layers, 2, batch_size, hidden_size) \
                 .transpose(1, 2).contiguous() \
                 .view(self.num_layers, batch_size, hidden_size * 2)
            return h

    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs):
        """Performs a single decoding step."""
        embedded = self.embedding(decoder_input)
        embedded = self.input_dropout(embedded)
        
        decoder_output, decoder_hidden = self.rnn(embedded, decoder_hidden)
        
        if self.attention:
            attn_output, _ = self.attention(decoder_output, encoder_outputs)
            logits = self.out_linear(attn_output)
        else:
            logits = self.out_linear(decoder_output)
            
        return logits, decoder_hidden

    def forward(self, decoder_inputs, encoder_hidden, encoder_outputs):
        """
        Forward pass for supervised training (Teacher Forcing).

        Args:
            decoder_inputs (torch.Tensor): Shifted ground truth programs.
                Shape: [B, S_prog]
            encoder_hidden: Encoder hidden state.
            encoder_outputs (torch.Tensor): Encoder outputs.
                Shape: [B, S_quest, H_enc]

        Returns:
            torch.Tensor: Logits for each token. Shape: [B, S_prog, V]
        """
        decoder_hidden = self._format_encoder_hidden(encoder_hidden)
        
        embedded = self.embedding(decoder_inputs)
        embedded = self.input_dropout(embedded)
        
        decoder_outputs, _ = self.rnn(embedded, decoder_hidden)
        
        if self.attention:
            attn_outputs, _ = self.attention(decoder_outputs, encoder_outputs)
            logits = self.out_linear(attn_outputs)
        else:
            logits = self.out_linear(decoder_outputs)
            
        return logits, decoder_hidden

    def forward_sample(self, encoder_hidden, encoder_outputs, reinforce_sample=False):
        """
        Forward pass for inference/sampling (REINFORCE or greedy).

        Args:
            encoder_hidden: Encoder hidden state.
            encoder_outputs (torch.Tensor): Encoder outputs.
                Shape: [B, S_quest, H_enc]
            reinforce_sample (bool): Whether to sample from the distribution
                (for REINFORCE) or take the argmax (for greedy).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - output_symbols: Sampled program indices. Shape: [B, max_len]
            - output_logprobs: Log probabilities of sampled tokens. Shape: [B, max_len]
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        decoder_hidden = self._format_encoder_hidden(encoder_hidden)
        
        # Initialize with <START> token
        decoder_input = torch.tensor([self.start_id] * batch_size, 
                                     dtype=torch.long, device=device).unsqueeze(1)
        
        output_symbols = []
        output_logprobs = []

        for _ in range(self.max_seq_len):
            logits, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # logits shape: [B, 1, V]
            
            # Squeeze logits to [B, V]
            logits = logits.squeeze(1)
            log_prob = F.log_softmax(logits, dim=1) # [B, V]
            
            if reinforce_sample:
                # Sample from the distribution
                dist = torch.distributions.Categorical(logits=logits)
                next_symbol = dist.sample() # [B]
                step_log_prob = dist.log_prob(next_symbol) # [B]
            else:
                # Greedy decoding
                next_symbol = torch.argmax(log_prob, dim=1) # [B]
                step_log_prob = log_prob.gather(1, next_symbol.unsqueeze(-1)).squeeze(-1) # [B]
            
            output_symbols.append(next_symbol)
            output_logprobs.append(step_log_prob)
            
            # Use the sampled symbol as the next input
            decoder_input = next_symbol.unsqueeze(1) # [B, 1]
            
        # Stack all steps
        output_symbols = torch.stack(output_symbols, dim=1) # [B, max_len]
        output_logprobs = torch.stack(output_logprobs, dim=1) # [B, max_len]
        
        return output_symbols, output_logprobs


class LstmSeq2Seq(nn.Module):
    """
    Wrapper for the Encoder-Decoder LSTM model.
    Provides a clean API for forward and sampling passes.
    """
    def __init__(self, vocab, word_vec_dim, hidden_size, num_layers,
                 input_dropout_prob, rnn_dropout_prob,
                 bidirectional_encoder=True, use_attention=True):
        super(LstmSeq2Seq, self).__init__()
        
        q_vocab_size = len(vocab["question_token_to_idx"])
        q_max_len = 50 # Max question length
        
        self.encoder = LstmEncoder(
            vocab_size=q_vocab_size,
            max_seq_len=q_max_len,
            word_vec_dim=word_vec_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_dropout_prob=input_dropout_prob,
            rnn_dropout_prob=rnn_dropout_prob,
            bidirectional=bidirectional_encoder,
        )
        
        self.decoder = LstmDecoder(
            vocab=vocab,
            word_vec_dim=word_vec_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_dropout_prob=input_dropout_prob,
            rnn_dropout_prob=rnn_dropout_prob,
            bidirectional_encoder=bidirectional_encoder,
            use_attention=use_attention,
        )

    def forward(self, questions, programs):
        """Supervised training forward pass."""
        encoder_outputs, encoder_hidden = self.encoder(questions)
        logits, _ = self.decoder(programs, encoder_hidden, encoder_outputs)
        return logits

    def forward_sample(self, questions, reinforce_sample=False):
        """Inference/RL forward pass."""
        encoder_outputs, encoder_hidden = self.encoder(questions)
        symbols, logprobs = self.decoder.forward_sample(
            encoder_hidden, encoder_outputs, reinforce_sample
        )
        return symbols, logprobs
