"""
Script for preprocessing CLEVR question JSON files.

This script:
1. Loads raw CLEVR questions from a JSON file.
2. Builds a vocabulary for questions, programs, and answers.
3. Saves the vocabulary to a new JSON file (if specified).
4. Encodes the questions, programs, and answers into integer sequences.
5. Pads all sequences to a fixed maximum length.
6. Saves the processed, padded data into an H5 file.

Example (Train):
python scripts/preprocess_data.py \
    --input_json data/CLEVR_Dataset/Questions/CLEVR_train_questions.json \
    --output_h5 data/dataH5Files/clevr_train_questions.h5 \
    --output_vocab_json data/dataH5Files/clevr_vocab.json \
    --max_question_len 45 \
    --max_program_len 30

Example (Validation):
python scripts/preprocess_data.py \
    --input_json data/CLEVR_Dataset/Questions/CLEVR_val_questions.json \
    --input_vocab_json data/dataH5Files/clevr_vocab.json \
    --output_h5 data/dataH5Files/clevr_val_questions.h5 \
    --allow_unk 1
"""

import argparse
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

import src.config as config
from src.utils.logger import setup_logger, log
from src.utils.program_utils import program_to_str
from src.vocabulary import (
    build_vocab,
    tokenize,
    encode,
    SPECIAL_TOKENS
)


def load_questions(json_path: Path) -> list:
    """Loads the 'questions' list from the input JSON file."""
    log.info(f"Loading questions from {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data['questions']
    except (IOError, json.JSONDecodeError, KeyError) as e:
        log.error(f"Failed to load or parse {json_path}: {e}")
        raise


def build_all_vocabs(questions: list, min_token_count: int, mode: str) -> dict:
    """Builds vocabs for questions, programs, and answers."""
    log.info("Building vocabularies...")
    
    # 1. Question vocab
    q_punct_keep = [';', ',']
    q_punct_remove = ['?', '.']
    q_tokenizer_kwargs = {'punct_to_keep': q_punct_keep, 'punct_to_remove': q_punct_remove}
    question_token_to_idx = build_vocab(
        (q['question'] for q in questions),
        min_token_count=min_token_count,
        **q_tokenizer_kwargs
    )
    log.info(f"Built question vocab with {len(question_token_to_idx)} tokens.")
    
    # 2. Program vocab
    all_program_strs = []
    for q in questions:
        if 'program' not in q:
            continue
        program_str = program_to_str(q['program'], mode)
        if program_str is not None:
            all_program_strs.append(program_str)
    program_token_to_idx = build_vocab(all_program_strs)
    log.info(f"Built program vocab with {len(program_token_to_idx)} tokens.")

    # 3. Answer vocab
    answer_token_to_idx = build_vocab(
        (q['answer'] for q in questions if 'answer' in q)
    )
    log.info(f"Built answer vocab with {len(answer_token_to_idx)} tokens.")
    
    return {
        'question_token_to_idx': question_token_to_idx,
        'program_token_to_idx': program_token_to_idx,
        'answer_token_to_idx': answer_token_to_idx,
    }


def load_vocab_from_file(vocab_path: Path) -> dict:
    """Loads an existing vocabulary JSON file."""
    log.info(f"Loading existing vocabulary from {vocab_path}")
    try:
        with open(vocab_path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        log.error(f"Failed to load vocab {vocab_path}: {e}")
        raise


def encode_and_pad_data(questions, vocab, max_q_len, max_p_len, mode, allow_unk):
    """Encodes all data, pads it, and prepares it for H5 storage."""
    log.info("Encoding and padding data...")
    
    null_q_token = SPECIAL_TOKENS['<NULL>']
    null_p_token = SPECIAL_TOKENS['<NULL>']
    
    q_punct_keep = [';', ',']
    q_punct_remove = ['?', '.']
    
    encoded_questions = []
    encoded_programs = []
    encoded_answers = []
    image_indices = []
    orig_indices = []

    for orig_idx, q in enumerate(tqdm(questions, desc="Encoding sequences")):
        # --- Question ---
        q_tokens = tokenize(
            q['question'],
            punct_to_keep=q_punct_keep,
            punct_to_remove=q_punct_remove
        )
        q_encoded = encode(q_tokens, vocab['question_token_to_idx'], allow_unk)
        # Pad
        q_encoded.extend([null_q_token] * (max_q_len - len(q_encoded)))
        encoded_questions.append(q_encoded[:max_q_len]) # Truncate if too long

        # --- Program ---
        if 'program' in q:
            program_str = program_to_str(q['program'], mode)
            p_tokens = tokenize(program_str)
            p_encoded = encode(p_tokens, vocab['program_token_to_idx'], allow_unk)
            # Pad
            p_encoded.extend([null_p_token] * (max_p_len - len(p_encoded)))
            encoded_programs.append(p_encoded[:max_p_len]) # Truncate

        # --- Answer ---
        if 'answer' in q:
            a_idx = vocab['answer_token_to_idx'].get(q['answer'])
            if a_idx is not None:
                encoded_answers.append(a_idx)
            elif allow_unk:
                encoded_answers.append(SPECIAL_TOKENS['<UNK>'])
            else:
                log.error(f"Unknown answer token: {q['answer']}")
                encoded_answers.append(-1) # Error
        
        image_indices.append(q['image_index'])
        orig_indices.append(orig_idx)

    log.info("Encoding complete.")
    
    return {
        'questions': np.asarray(encoded_questions, dtype=np.int32),
        'programs': np.asarray(encoded_programs, dtype=np.int32),
        'answers': np.asarray(encoded_answers, dtype=np.int32),
        'image_idxs': np.asarray(image_indices, dtype=np.int32),
        'orig_idxs': np.asarray(orig_indices, dtype=np.int32),
    }


def save_h5(output_path: Path, data: dict):
    """Saves the processed data to an H5 file."""
    log.info(f"Saving processed data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('questions', data=data['questions'])
            f.create_dataset('image_idxs', data=data['image_idxs'])
            f.create_dataset('orig_idxs', data=data['orig_idxs'])
            
            if len(data['programs']) > 0:
                f.create_dataset('programs', data=data['programs'])
            if len(data['answers']) > 0:
                f.create_dataset('answers', data=data['answers'])
        log.info("H5 file saved successfully.")
    except (IOError, h5py.error) as e:
        log.error(f"Failed to save H5 file: {e}")
        raise


def main(args):
    setup_logger(config.LOG_FILE)
    
    questions = load_questions(args.input_json)
    
    # --- 1. Load or Build Vocab ---
    if args.input_vocab_json:
        # Use existing vocab (for val/test)
        vocab = load_vocab_from_file(args.input_vocab_json)
    else:
        # Build new vocab (for train)
        vocab = build_all_vocabs(questions, args.min_token_count, args.mode)
        if args.output_vocab_json:
            log.info(f"Saving vocabulary to {args.output_vocab_json}...")
            args.output_vocab_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_vocab_json, 'w') as f:
                json.dump(vocab, f, indent=4)
        else:
            log.warning("No --output_vocab_json specified. Vocab will not be saved.")
            
    # --- 2. Encode and Pad Data ---
    encoded_data = encode_and_pad_data(
        questions,
        vocab,
        args.max_question_len,
        args.max_program_len,
        args.mode,
        args.allow_unk == 1
    )

    # --- 3. Save to H5 ---
    save_h5(args.output_h5, encoded_data)
    
    log.info(f"Preprocessing complete for {args.input_json}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CLEVR question data.")
    
    # --- I/O Paths ---
    parser.add_argument('--input_json', type=Path, required=True,
                        help="Input CLEVR questions JSON file.")
    parser.add_argument('--output_h5', type=Path, required=True,
                        help="Output H5 file for processed data.")
    parser.add_argument('--input_vocab_json', type=Path, default=None,
                        help="Path to an existing vocabulary JSON. (For val/test)")
    parser.add_argument('--output_vocab_json', type=Path, default=None,
                        help="Path to save the newly built vocabulary. (For train)")

    # --- Preprocessing ---
    parser.add_argument('--mode', default='prefix', choices=['chain', 'prefix', 'postfix'],
                        help="Program representation mode.")
    parser.add_argument('--min_token_count', type=int, default=config.MIN_TOKEN_COUNT,
                        help="Minimum token count to be included in vocab.")
    parser.add_argument('--allow_unk', type=int, default=0, choices=[0, 1],
                        help="Whether to allow <UNK> tokens (1) or error (0).")
    
    # --- Padding ---
    parser.add_argument('--max_question_len', type=int, default=config.MAX_QUESTION_LEN,
                        help="Maximum length for question sequences.")
    parser.add_argument('--max_program_len', type=int, default=config.MAX_PROGRAM_LEN,
                        help="Maximum length for program sequences.")

    args = parser.parse_args()
    main(args)
