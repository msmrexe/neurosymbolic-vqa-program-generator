import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from src.utils.logger import log
from src.evaluation.eval_model import evaluate_model


class TrainerReinforce:
    """
    Handles the REINFORCE fine-tuning loop for a Seq2Seq model.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        executor,
        vocab,
        device,
        learning_rate=1e-5,
        num_iters=100000,
        reward_decay=0.99,
        log_interval=100,
        val_interval=1000,
        model_save_path="models/reinforce_model.pth",
    ):
        """
        Initializes the REINFORCE Trainer.

        Args:
            model (nn.Module): The pre-trained Seq2Seq model.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            executor (ClevrExecutor): The symbolic program executor.
            vocab (dict): The loaded vocabulary.
            device (torch.device): The device to run training on.
            learning_rate (float): The learning rate for the optimizer.
            num_iters (int): Total number of training iterations.
            reward_decay (float): Decay factor for the reward baseline.
            log_interval (int): How often to log training metrics.
            val_interval (int): How often to run validation.
            model_save_path (str): Path to save the best model.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.executor = executor
        self.vocab = vocab
        self.device = device

        # Hyperparameters
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.model_save_path = model_save_path

        # Token indices
        self.pad_idx = vocab["program_token_to_idx"]["<NULL>"]

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.baseline = 0.0  # Moving average of rewards
        self.best_val_accuracy = -1.0
        
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def _get_batch_reward(self, predicted_programs, gt_answers, image_indices, split):
        """
        Calculates the reward for a batch of generated programs.
        Reward is 1 if the executed program yields the correct answer, 0 otherwise.
        
        Args:
            predicted_programs (Tensor): [B, S] tensor of sampled program indices.
            gt_answers (Tensor): [B] tensor of ground truth answer indices.
            image_indices (Tensor): [B] tensor of image indices.
            split (str): The dataset split ('train' or 'val').
            
        Returns:
            Tensor: [B] tensor of rewards (1.0 or 0.0).
        """
        batch_size = predicted_programs.size(0)
        rewards = torch.zeros(batch_size, device=self.device)
        
        # Get ground truth answer strings
        gt_answer_strs = [
            self.vocab["answer_idx_to_token"][idx.item()] 
            for idx in gt_answers
        ]

        for i in range(batch_size):
            program_indices = predicted_programs[i].cpu().numpy()
            image_idx = image_indices[i].item()
            
            # Run the executor
            pred_answer_str = self.executor.run(
                program_indices, image_idx, split=split
            )
            
            # Compare and assign reward
            if pred_answer_str == gt_answer_strs[i]:
                rewards[i] = 1.0
        
        return rewards

    def _run_train_epoch(self):
        """Runs a single training epoch using REINFORCE."""
        self.model.train()  # Set model to training mode
        loop = tqdm(self.train_loader, desc=f"Epoch {self.epoch} [REINFORCE]")

        for batch in loop:
            if self.global_step >= self.num_iters:
                log.info("Reached target iterations. Stopping training.")
                return False  # Signal to stop training

            # Unpack batch and move to device
            questions, _, gt_answers, image_indices = batch
            questions = questions.to(self.device)
            gt_answers = gt_answers.to(self.device)
            image_indices = image_indices.to(self.device)

            self.optimizer.zero_grad()

            # --- Forward Pass (Sampling) ---
            # Sample programs and get their log probabilities
            # reinforce_sample=True enables sampling (e.g., from Categorical)
            predicted_programs, log_probs = self.model.forward_sample(
                questions, reinforce_sample=True
            )
            # log_probs shape: [B, S]

            # --- Calculate Rewards ---
            rewards = self._get_batch_reward(
                predicted_programs, gt_answers, image_indices, "train"
            )
            
            # --- Update Baseline ---
            # Use a moving average of the batch reward
            batch_reward = rewards.mean().item()
            self.baseline = (batch_reward * (1 - self.reward_decay)) + \
                            (self.baseline * self.reward_decay)
            
            # --- Calculate Advantage ---
            # Advantage = (Reward - Baseline)
            # We detach baseline so no gradients flow through it
            advantage = rewards - self.baseline

            # --- Calculate REINFORCE Loss ---
            # Loss = - (sum of log_probs_for_action * advantage)
            
            # Mask out padding tokens from the loss
            mask = (predicted_programs != self.pad_idx).float()
            masked_log_probs = log_probs * mask
            
            # Sum log_probs for each program
            sum_log_probs = masked_log_probs.sum(dim=1)  # [B]

            # Policy gradient loss
            # Unsqueeze advantage to [B, 1] for broadcasting (or just multiply [B] * [B])
            reinforce_loss = -(sum_log_probs * advantage.detach())
            
            # Average loss over the batch
            loss = reinforce_loss.mean()

            # --- Backward Pass ---
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
            batch_accuracy = rewards.mean().item()

            # --- Logging ---
            if self.global_step % self.log_interval == 0:
                log.info(
                    f"[Step {self.global_step}/{self.num_iters}] "
                    f"Loss: {loss.item():.4f}, "
                    f"Reward (Acc): {batch_accuracy:.4f}, "
                    f"Baseline: {self.baseline:.4f}"
                )
                loop.set_postfix(
                    loss=loss.item(), 
                    reward=batch_accuracy,
                    baseline=self.baseline
                )

            # --- Validation ---
            if self.global_step % self.val_interval == 0:
                self._run_validation()
                
        return True # Signal to continue training

    def _run_validation(self):
        """Runs validation and saves the model if it's the best one."""
        log.info(f"--- Running validation at step {self.global_step} ---")
        # Use the dedicated evaluation function
        # This function internally uses greedy sampling (reinforce_sample=False)
        val_accuracy = evaluate_model(
            self.model,
            self.val_loader,
            self.executor,
            self.vocab,
            self.device,
            split="val",
        )
        
        log.info(f"Validation Accuracy: {val_accuracy*100:.2f}%")

        if val_accuracy > self.best_val_accuracy:
            log.info(f"New best validation accuracy! Saving model to {self.model_save_path}")
            self.best_val_accuracy = val_accuracy
            torch.save(self.model.state_dict(), self.model_save_path)
        
        # Set model back to training mode
        self.model.train()

    def train(self):
        """Main training entry point."""
        log.info("Starting REINFORCE fine-tuning...")
        log.info(f"  Device: {self.device}")
        log.info(f"  Num iterations: {self.num_iters}")
        log.info(f"  Model save path: {self.model_save_path}")

        while self.global_step < self.num_iters:
            self.epoch += 1
            log.info(f"--- Starting Epoch {self.epoch} ---")
            
            should_continue = self._run_train_epoch()
            if not should_continue:
                break

        log.info("REINFORCE training complete.")
        log.info(f"Best validation accuracy: {self.best_val_accuracy*100:.2f}%")
        log.info(f"Best model saved to {self.model_save_path}")
