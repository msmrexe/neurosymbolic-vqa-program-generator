"""
Main training script for the Neurosymbolic VQA model.

This script handles both supervised pre-training and REINFORCE fine-tuning
for both LSTM and Transformer models.

Example (Supervised LSTM):
python scripts/train.py \
    --model_type lstm \
    --train_mode supervised \
    --model_save_path models/supervised_lstm.pth \
    --num_iters 100000

Example (REINFORCE Transformer):
python scripts/train.py \
    --model_type transformer \
    --train_mode reinforce \
    --load_model models/supervised_transformer.pth \
    --model_save_path models/reinforce_transformer.pth \
    --num_iters 50000 \
    --learning_rate 1e-5
"""

import argparse
import torch
from pathlib import Path

import src.config as config
from src.utils.logger import setup_logger, log
from src.vocabulary import load_vocab
from src.executor import ClevrExecutor
from src.data_loader import get_dataloader
from src.models import LstmSeq2Seq, TransformerSeq2Seq
from src.training import TrainerSupervised, TrainerReinforce


def get_model(args, vocab):
    """Instantiates and returns the correct model based on args."""
    log.info(f"Initializing model: {args.model_type}")
    
    if args.model_type == 'lstm':
        model = LstmSeq2Seq(
            vocab=vocab,
            word_vec_dim=args.lstm_word_vec_dim,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_num_layers,
            input_dropout_prob=args.lstm_input_dropout,
            rnn_dropout_prob=args.lstm_rnn_dropout,
            bidirectional_encoder=True,
            use_attention=True
        )
    elif args.model_type == 'transformer':
        model = TransformerSeq2Seq(
            vocab=vocab,
            d_model=args.tr_d_model,
            nhead=args.tr_nhead,
            num_encoder_layers=args.tr_num_encoder_layers,
            num_decoder_layers=args.tr_num_decoder_layers,
            dim_feedforward=args.tr_dim_feedforward,
            dropout=args.tr_dropout,
            max_seq_len=max(config.MAX_QUESTION_LEN, config.MAX_PROGRAM_LEN)
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
        
    return model


def get_trainer(args, model, train_loader, val_loader, executor, vocab, device):
    """Instantiates and returns the correct trainer based on args."""
    log.info(f"Initializing trainer: {args.train_mode}")
    
    common_kwargs = {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "executor": executor,
        "vocab": vocab,
        "device": device,
        "num_iters": args.num_iters,
        "log_interval": args.log_interval,
        "val_interval": args.val_interval,
        "model_save_path": args.model_save_path
    }
    
    if args.train_mode == 'supervised':
        trainer = TrainerSupervised(
            learning_rate=args.learning_rate,
            **common_kwargs
        )
    elif args.train_mode == 'reinforce':
        trainer = TrainerReinforce(
            learning_rate=args.learning_rate,
            reward_decay=args.reward_decay,
            **common_kwargs
        )
    else:
        raise ValueError(f"Unknown train_mode: {args.train_mode}")
        
    return trainer


def main(args):
    # --- Setup ---
    setup_logger(config.LOG_FILE)
    torch.manual_seed(1234)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        log.warning("CUDA not available. Falling back to CPU.")
        args.device = 'cpu'
    device = torch.device(args.device)
    log.info(f"Using device: {device}")

    # --- Load Data and Utils ---
    log.info("Loading vocab, executor, and data loaders...")
    vocab = load_vocab(config.VOCAB_JSON_FILE)
    executor = ClevrExecutor(
        train_scene_json=config.TRAIN_SCENES_JSON,
        val_scene_json=config.VAL_SCENES_JSON,
        vocab_json=config.VOCAB_JSON_FILE
    )
    
    train_loader = get_dataloader(
        h5_path=config.TRAIN_H5_FILE,
        vocab_json_path=config.VOCAB_JSON_FILE,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = get_dataloader(
        h5_path=config.VAL_H5_FILE,
        vocab_json_path=config.VOCAB_JSON_FILE,
        batch_size=args.batch_size,
        shuffle=False  # No need to shuffle validation
    )

    # --- Initialize Model ---
    model = get_model(args, vocab)
    model.to(device)
    
    # --- Load Pretrained Weights (if specified) ---
    if args.load_model:
        if args.load_model.exists():
            log.info(f"Loading pretrained weights from: {args.load_model}")
            model.load_state_dict(torch.load(args.load_model, map_location=device))
        else:
            log.error(f"Pretrained model path not found: {args.load_model}")
            raise FileNotFoundError(f"File not found: {args.load_model}")
    elif args.train_mode == 'reinforce':
        log.warning("Training in REINFORCE mode without a --load_model path. "
                    "This is only recommended for testing.")

    # --- Initialize Trainer and Start Training ---
    log.info(f"Starting training run: {args.model_type} ({args.train_mode})")
    trainer = get_trainer(args, model, train_loader, val_loader, executor, vocab, device)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        log.info("Training interrupted by user. Exiting.")
    except Exception as e:
        log.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        
    log.info("Training run finished.")


def populate_args_from_config(parser):
    """Adds arguments from the config file to the parser."""
    
    # --- General Training ---
    parser.add_argument('--num_iters', type=int, default=config.NUM_ITERS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--reward_decay', type=float, default=config.REWARD_DECAY)
    parser.add_argument('--device', type=str, default=config.DEVICE, choices=['cuda', 'cpu'])
    parser.add_argument('--log_interval', type=int, default=config.LOG_INTERVAL)
    parser.add_argument('--val_interval', type=int, default=config.VAL_INTERVAL)
    
    # --- LSTM Config ---
    parser.add_argument('--lstm_word_vec_dim', type=int, default=config.LSTM_WORD_VEC_DIM)
    parser.add_argument('--lstm_hidden_size', type=int, default=config.LSTM_HIDDEN_SIZE)
    parser.add_argument('--lstm_num_layers', type=int, default=config.LSTM_NUM_LAYERS)
    parser.add_argument('--lstm_rnn_dropout', type=float, default=config.LSTM_RNN_DROPOUT)
    parser.add_argument('--lstm_input_dropout', type=float, default=config.LSTM_INPUT_DROPOUT)
    
    # --- Transformer Config ---
    parser.add_argument('--tr_d_model', type=int, default=config.TR_D_MODEL)
    parser.add_argument('--tr_nhead', type=int, default=config.TR_NHEAD)
    parser.add_argument('--tr_num_encoder_layers', type=int, default=config.TR_NUM_ENCODER_LAYERS)
    parser.add_argument('--tr_num_decoder_layers', type=int, default=config.TR_NUM_DECODER_LAYERS)
    parser.add_argument('--tr_dim_feedforward', type=int, default=config.TR_DIM_FEEDFORWARD)
    parser.add_argument('--tr_dropout', type=float, default=config.TR_DROPOUT)
    
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Neurosymbolic VQA model.")
    
    # --- Core Arguments ---
    parser.add_argument('--model_type', type=str, required=True, choices=['lstm', 'transformer'],
                        help="The type of model architecture to train.")
    parser.add_argument('--train_mode', type=str, required=True, choices=['supervised', 'reinforce'],
                        help="The training paradigm to use.")
    
    # --- Paths ---
    parser.add_argument('--load_model', type=Path, default=None,
                        help="Path to a pretrained model checkpoint to load. (Required for REINFORCE)")
    parser.add_argument('--model_save_path', type=Path, default=Path(config.MODEL_DIR, "model.pth"),
                        help="Path to save the best trained model.")
                        
    # --- Add all hyperparameters from config ---
    parser = populate_args_from_config(parser)
    
    args = parser.parse_args()
    main(args)
