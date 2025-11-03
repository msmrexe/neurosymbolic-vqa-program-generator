"""
Central configuration file for all paths, hyperparameters, and model settings.
"""
from pathlib import Path

# --- Base Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

# --- Data Paths ---
CLEVR_DIR = DATA_DIR / "CLEVR_Dataset"
H5_DIR = DATA_DIR / "dataH5Files"

# Input JSON files
TRAIN_QUESTIONS_JSON = CLEVR_DIR / "Questions" / "CLEVR_train_questions.json"
VAL_QUESTIONS_JSON = CLEVR_DIR / "Questions" / "CLEVR_val_questions.json"
TEST_QUESTIONS_JSON = CLEVR_DIR / "Questions" / "CLEVR_test_questions.json"

TRAIN_SCENES_JSON = CLEVR_DIR / "Scenes" / "CLEVR_train_scenes.json"
VAL_SCENES_JSON = CLEVR_DIR / "Scenes" / "CLEVR_val_scenes.json"
TEST_SCENES_JSON = CLEVR_DIR / "Scenes" / "CLEVR_test_scenes.json"

# Output H5 and Vocab files
TRAIN_H5_FILE = H5_DIR / "clevr_train_questions.h5"
VAL_H5_FILE = H5_DIR / "clevr_val_questions.h5"
TEST_H5_FILE = H5_DIR / "clevr_test_questions.h5"
VOCAB_JSON_FILE = H5_DIR / "clevr_vocab.json"

# --- Logger ---
LOG_FILE = LOG_DIR / "program_generator.log"

# --- Data Config ---
MAX_QUESTION_LEN = 45  # Max length of a question
MAX_PROGRAM_LEN = 30   # Max length of a program
MIN_TOKEN_COUNT = 1    # For building vocabulary

# --- Model Hyperparameters ---
BATCH_SIZE = 64
NUM_ITERS = 100000
LEARNING_RATE = 1e-4
REWARD_DECAY = 0.99  # For REINFORCE baseline

# LSTM-specific
LSTM_WORD_VEC_DIM = 256
LSTM_HIDDEN_SIZE = 512
LSTM_NUM_LAYERS = 2
LSTM_RNN_DROPOUT = 0.1
LSTM_INPUT_DROPOUT = 0.1

# Transformer-specific
TR_D_MODEL = 256
TR_NHEAD = 8
TR_NUM_ENCODER_LAYERS = 3
TR_NUM_DECODER_LAYERS = 3
TR_DIM_FEEDFORWARD = 1024
TR_DROPOUT = 0.1

# --- Training Loop Config ---
LOG_INTERVAL = 100
VAL_INTERVAL = 1000
DEVICE = "cuda"  # "cuda" or "cpu"

# --- ICL / LLM Config ---
LLM_MODEL_ID = "unsloth/Llama-3.2-3B-Instruct"
ICL_NUM_TEST_SAMPLES = 100  # How many test samples to run for ICL eval
ICL_SHOTS_LIST = [0, 2, 5, 10]  # List of few-shot examples to test
