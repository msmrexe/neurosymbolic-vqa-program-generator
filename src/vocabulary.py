import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Iterable

from src.utils.logger import log

# Define special tokens
SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}


def _invert_dict(d: dict) -> dict:
    """Inverts a dictionary, mapping values to keys."""
    return {v: k for k, v in d.items()}


def load_vocab(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads a vocabulary JSON file.
    
    Also creates inverted mappings (idx_to_token) for convenience.
    
    Args:
        path (str): Path to the vocabulary JSON file.
        
    Returns:
        Dict: The loaded vocabulary dictionary.
    """
    log.info(f"Loading vocabulary from {path}...")
    p = Path(path)
    if not p.exists():
        log.error(f"Vocabulary file not found: {path}")
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
        
    with open(p, 'r') as f:
        vocab = json.load(f)
    
    # Create inverted mappings
    try:
        vocab['question_idx_to_token'] = _invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = _invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = _invert_dict(vocab['answer_token_to_idx'])
    except KeyError as e:
        log.error(f"Vocabulary file is missing expected key: {e}")
        raise
    
    # Sanity checks
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    
    log.info("Vocabulary loaded successfully.")
    return vocab


def tokenize(
    s: str,
    delim: str = ' ',
    add_start_token: bool = True,
    add_end_token: bool = True,
    punct_to_keep: Optional[List[str]] = None,
    punct_to_remove: Optional[List[str]] = None
) -> List[str]:
    """
    Tokenizes a string sequence.
    
    Args:
        s (str): The input string.
        delim (str): The delimiter to split on.
        add_start_token (bool): Whether to prepend <START>.
        add_end_token (bool): Whether to append <END>.
        punct_to_keep (list, optional): Punctuation to keep as separate tokens.
        punct_to_remove (list, optional): Punctuation to remove entirely.
        
    Returns:
        List[str]: A list of string tokens.
    """
    if punct_to_keep:
        # Add space around punctuation to keep
        for p in punct_to_keep:
            s = s.replace(p, f'{delim}{p}{delim}')
            
    if punct_to_remove:
        # Remove punctuation
        for p in punct_to_remove:
            s = s.replace(p, '')
            
    # Split on the delimiter and remove empty strings
    tokens = [token for token in s.split(delim) if token]
    
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
        
    return tokens


def build_vocab(
    sequences: Iterable[str],
    min_token_count: int = 1,
    delim: str = ' ',
    punct_to_keep: Optional[List[str]] = None,
    punct_to_remove: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Builds a vocabulary (token_to_idx mapping) from an iterable of sequences.
    
    Args:
        sequences (Iterable[str]): An iterable of strings.
        min_token_count (int): Minimum count for a token to be included.
        ... (other args passed to tokenize)
        
    Returns:
        Dict[str, int]: A dictionary mapping token strings to integer indices.
    """
    token_counts = Counter()
    
    # Tokenize arguments, but without start/end tokens for vocab building
    tokenize_kwargs = {
        'delim': delim,
        'punct_to_keep': punct_to_keep,
        'punct_to_remove': punct_to_remove,
        'add_start_token': False,
        'add_end_token': False,
    }
    
    for seq in sequences:
        seq_tokens = tokenize(seq, **tokenize_kwargs)
        token_counts.update(seq_tokens)
        
    # Start with special tokens
    token_to_idx = dict(SPECIAL_TOKENS)
    
    # Sort tokens by frequency (most common first) for consistent indexing
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    for token, count in sorted_tokens:
        if count >= min_token_count:
            if token not in token_to_idx:
                token_to_idx[token] = len(token_to_idx)
                
    return token_to_idx


def encode(
    seq_tokens: List[str],
    token_to_idx: Dict[str, int],
    allow_unk: bool = False
) -> List[int]:
    """
    Encodes a list of token strings into a list of integer indices.
    
    Args:
        seq_tokens (List[str]): The list of token strings.
        token_to_idx (Dict[str, int]): The vocabulary mapping.
        allow_unk (bool): If True, map unknown tokens to <UNK>.
                          If False, raise a KeyError.
                          
    Returns:
        List[int]: A list of integer indices.
    """
    seq_idx = []
    for token in seq_tokens:
        idx = token_to_idx.get(token)
        if idx is not None:
            seq_idx.append(idx)
        elif allow_unk:
            seq_idx.append(token_to_idx['<UNK>'])
        else:
            raise KeyError(f'Token "{token}" not in vocab and allow_unk=False')
            
    return seq_idx


def decode(
    seq_idx: List[int],
    idx_to_token: Dict[int, str],
    delim: Optional[str] = ' ',
    stop_at_end: bool = True
) -> Any:
    """
    Decodes a list of integer indices back into token strings.
    
    Args:
        seq_idx (List[int]): The list of integer indices.
        idx_to_token (Dict[int, str]): The inverted vocabulary mapping.
        delim (str, optional): If provided, join tokens with this delimiter.
                               If None, return a list of tokens.
        stop_at_end (bool): If True, stop decoding after the first <END> token.
        
    Returns:
        Union[List[str], str]: A list of tokens or a single string.
    """
    tokens = []
    for idx in seq_idx:
        token = idx_to_token.get(idx, '<UNK>')
        if stop_at_end and token == '<END>':
            break
        if token != '<NULL>':
            tokens.append(token)
            
    return delim.join(tokens) if delim is not None else tokens
