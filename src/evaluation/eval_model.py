import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

from src.executor import ClevrExecutor
from src.utils.logger import log


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    executor: ClevrExecutor,
    vocab: dict,
    device: torch.device,
    split: str = "val",
):
    """
    Evaluates a trained Seq2Seq model (LSTM or Transformer) on a given dataset.

    It generates programs, executes them, and compares the resulting
    answer to the ground truth answer.

    Args:
        model (nn.Module): The trained model (e.g., LstmSeq2Seq).
        dataloader (DataLoader): DataLoader for the validation or test set.
        executor (ClevrExecutor): The symbolic program executor.
        vocab (dict): The loaded vocabulary file.
        device (torch.device): The device to run evaluation on (e.g., 'cuda').
        split (str): The dataset split to use for the executor ('train' or 'val').

    Returns:
        float: The final accuracy (0.0 to 1.0).
    """
    log.info(f"Starting evaluation on '{split}' split...")
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    # We don't need to compute gradients during evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating ({split})"):
            # Unpack batch and move to device
            questions, _, gt_answers, image_indices = batch
            questions = questions.to(device)
            gt_answers = gt_answers.to(device)

            # Generate programs using the model's sampling method
            # This returns [batch_size, max_len] tensor of program indices
            # and [batch_size, max_len] tensor of log probabilities
            predicted_programs, _ = model.forward_sample(
                questions, reinforce_sample=False
            )

            # Iterate over each sample in the batch
            for i in range(questions.size(0)):
                pred_program_indices = predicted_programs[i]
                image_idx = image_indices[i].item()
                gt_answer_idx = gt_answers[i].item()

                # Get the string representation of the ground truth answer
                gt_answer_str = vocab["answer_idx_to_token"].get(
                    gt_answer_idx, "<UNK>"
                )

                # Execute the generated program
                # The executor.run() expects a list/tensor of program token indices
                pred_answer_str = executor.run(
                    pred_program_indices.cpu().numpy(), image_idx, split=split
                )

                # Check for correctness
                if pred_answer_str == gt_answer_str:
                    total_correct += 1
                total_samples += 1

    if total_samples == 0:
        log.warning("No samples found in the evaluation dataloader.")
        return 0.0

    # Calculate final accuracy
    accuracy = total_correct / total_samples
    log.info(
        f"Evaluation complete: "
        f"Accuracy = {accuracy*100:.2f}% ({total_correct}/{total_samples})"
    )
    return accuracy
