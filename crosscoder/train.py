import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import wandb # Make sure wandb is installed and you're logged in via terminal (`wandb login`)
import tqdm
import math # For checking NaN/inf

class Trainer:
    """
    Orchestrates the training process for the CrossCoder model.
    Handles the training loop, optimization, learning rate scheduling,
    loss calculation, logging, and saving checkpoints.
    """
    def __init__(self, cfg: dict, crosscoder: CrossCoder, buffer: Buffer):
        """
        Initializes the Trainer.


        Args:
            cfg (dict): Configuration dictionary. Expects keys like:
                        'batch_size', 'num_tokens', 'lr', 'beta1', 'beta2',
                        'l1_coeff', 'wandb_project', 'wandb_entity', 'log_every',
                        'save_every', 'device'.
            crosscoder (CrossCoder): The CrossCoder model instance to train.
            buffer (Buffer): The initialized Buffer instance providing activation data.
        """
        self.cfg = cfg
        self.crosscoder = crosscoder
        self.buffer = buffer
        self.device = cfg["device"]


        # Calculate total training steps based on tokens and batch size
        self.total_steps = cfg["num_tokens"] // cfg["batch_size"]
        print(f"Total training steps calculated: {self.total_steps}")




        # --- Optimizer ---
        # Adam optimizer is commonly used for training neural networks
        self.optimizer = optim.Adam(
            self.crosscoder.parameters(), # Pass the model's learnable parameters
            lr=cfg["lr"],                 # Learning rate
            betas=(cfg["beta1"], cfg["beta2"]), # Adam specific hyperparameters
            # weight_decay=cfg.get("weight_decay", 0.0) # Optional: L2 regularization on weights
        )
        print(f"Initialized Adam optimizer with LR: {cfg['lr']}")




        # --- Learning Rate Scheduler ---
        # Defines how the learning rate changes during training.
        # Here: constant LR for 80% of steps, then linear decay to 0.
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda # lr_lambda is a method defined below
        )
        print("Initialized LR scheduler (constant then linear decay).")


        # --- Step Counter ---
        self.step_counter = 0


        # --- WandB Initialization ---
        # Connect to Weights & Biases for logging
        try:
            wandb.init(
                project=cfg["wandb_project"],
                entity=cfg["wandb_entity"],
                config=cfg, # Log the configuration used for this run
                name=cfg.get("wandb_run_name", None) # Optional: custom run name
            )
            print(f"WandB initialized for project '{cfg['wandb_project']}' entity '{cfg['wandb_entity']}'.")
            # Watch the model gradients and parameters (optional, can be verbose)
            # wandb.watch(self.crosscoder, log_freq=max(100, cfg['log_every']), log='all')
        except Exception as e:
            print(f"Error initializing WandB: {e}")
            print("WandB logging will be disabled.")
            # Optionally disable wandb if init fails
            # wandb.init(mode="disabled")




    def lr_lambda(self, step: int) -> float:
        """
        Learning rate schedule: Constant for 80% of training, then linear decay.




        Args:
            step (int): The current training step number.




        Returns:
            float: The learning rate multiplier for the current step.
        """
        warmup_steps = self.cfg.get("lr_warmup_steps", 0) # Optional warmup
        decay_start_step = int(self.cfg.get("lr_decay_start_fraction", 0.8) * self.total_steps)




        if step < warmup_steps:
            # Linear warmup from 0 to 1
            return float(step) / float(max(1, warmup_steps))
        elif step < decay_start_step:
            # Constant phase
            return 1.0
        else:
            # Linear decay phase
            steps_into_decay = step - decay_start_step
            total_decay_steps = self.total_steps - decay_start_step
            decay_factor = 1.0 - (steps_into_decay / total_decay_steps)
            # Ensure decay factor doesn't go below a minimum (e.g., 0)
            return max(0.0, decay_factor)




    def get_l1_coeff(self) -> float:
        """
        L1 coefficient schedule: Linear ramp-up, then constant.
        Helps the model focus on reconstruction first, then sparsity.




        Returns:
            float: The L1 coefficient for the current step.
        """
        l1_warmup_frac = self.cfg.get("l1_warmup_fraction", 0.05) # Fraction of total steps for ramp-up
        l1_warmup_steps = int(l1_warmup_frac * self.total_steps)
        target_l1_coeff = self.cfg["l1_coeff"]




        if self.step_counter < l1_warmup_steps:
            # Linear ramp-up from 0 to target_l1_coeff
            return target_l1_coeff * (self.step_counter / max(1, l1_warmup_steps))
        else:
            # Constant phase at the target coefficient
            return target_l1_coeff


    def step(self) -> dict:
        """
        Performs a single training step:
        1. Get data batch from buffer.
        2. Run forward pass and calculate losses.
        3. Run backward pass to get gradients.
        4. Clip gradients.
        5. Update model weights using optimizer.
        6. Update learning rate using scheduler.
        7. Zero gradients.
        8. Return loss dictionary for logging.




        Returns:
            dict: Dictionary containing loss values and other metrics for this step.
        """
        # Set model to training mode (important for dropout, batchnorm layers, though not used here)
        self.crosscoder.train() # Set model to training mode
        acts_input = self.buffer.next()


        # Pass is_training=True to get_losses, which will pass it to encode
        losses: LossOutput = self.crosscoder.get_losses(acts_input, is_training=True)


        # For BatchTopK, the loss is primarily L2. L1_coeff is not used for the main loss.
        if self.cfg.get("sparsity_type", "l1") == "batch_top_k":
            loss = losses.l2_loss
            # TODO: Add L_aux if implementing the full Bussmann et al. auxiliary loss
            # For now, L_aux is effectively 0
            current_l1_coeff = 0.0 # L1 coeff is not part of BatchTopK primary loss
        else: # Original L1 sparsity
            current_l1_coeff = self.get_l1_coeff()
            loss = losses.l2_loss + current_l1_coeff * losses.l1_loss




        # Check for NaN/inf loss, which indicates instability
        if not torch.isfinite(loss):
              print(f"\nWarning: Non-finite loss detected at step {self.step_counter}: {loss.item()}")
              print(f"  L2: {losses.l2_loss.item()}, L1: {losses.l1_loss.item()}, L1 Coeff: {current_l1_coeff}")
              # Optional: Add more debugging here, e.g., check gradients later
              # For now, just return the problematic losses without backprop
              loss_dict = {
                  "loss": loss.item(), # Log the NaN/inf
                  "l2_loss": losses.l2_loss.item(),
                  "l1_loss": losses.l1_loss.item(),
                  "l0_loss": losses.l0_loss.item(),
                  "l1_coeff": current_l1_coeff,
                  "lr": self.scheduler.get_last_lr()[0],
                  "explained_variance": losses.explained_variance.item(),
                  "explained_variance_A": losses.explained_variance_A.item(),
                  "explained_variance_B": losses.explained_variance_B.item(),
              }
              self.step_counter += 1 # Still increment step counter
              return loss_dict # Skip backprop and optimizer step








        # 3. Backward pass (calculate gradients)
        # Gradients are calculated for the combined loss
        loss.backward()




        # 4. Clip gradients (prevent exploding gradients)
        # max_norm=1.0 is a common value
        grad_norm = clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)




        # 5. Optimizer step (update weights)
        self.optimizer.step()




        # 6. Scheduler step (update learning rate)
        self.scheduler.step()




        # 7. Zero gradients (clear gradients for the next iteration)
        self.optimizer.zero_grad()




        # 8. Prepare loss dictionary for logging
        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": current_l1_coeff,
            "lr": self.scheduler.get_last_lr()[0],
            "explained_variance": losses.explained_variance.item(),
            "explained_variance_A": losses.explained_variance_A.item(),
            "explained_variance_B": losses.explained_variance_B.item(),
            "grad_norm": grad_norm.item() # Log gradient norm after clipping
        }




        self.step_counter += 1
        return loss_dict




    def log(self, loss_dict: dict):
        """Logs the metrics dictionary to WandB and prints it."""
        # Log to WandB if initialized
        if wandb.run is not None:
            wandb.log(loss_dict, step=self.step_counter)




        # Print metrics locally (optional, can be verbose)
        log_str = f"Step: {self.step_counter:>7} | "
        log_str += f"Loss: {loss_dict['loss']:.4f} | "
        log_str += f"L2: {loss_dict['l2_loss']:.4f} | "
        log_str += f"L1: {loss_dict['l1_loss']:.4f} | "
        log_str += f"L0: {loss_dict['l0_loss']:.2f} | "
        log_str += f"ExplVar: {loss_dict['explained_variance']:.3f} | "
        log_str += f"LR: {loss_dict['lr']:.2e} | "
        log_str += f"L1Coeff: {loss_dict['l1_coeff']:.2f}"
        # Avoid printing every single step if logging frequently
        if self.step_counter % (self.cfg['log_every'] * 10) == 0 or self.step_counter <= 10:
              print(log_str)




    def save(self):
        """Saves a checkpoint of the CrossCoder model."""
        print(f"\nSaving checkpoint at step {self.step_counter}...")
        self.crosscoder.save(checkpoint_dir_str=self.cfg.get("checkpoint_path", "./checkpoints"))




    def train(self):
        """Runs the main training loop."""
        print("\n--- Starting Training ---")
        self.step_counter = 0 # Reset step counter at the beginning of training




        try:
            # Use tqdm for a progress bar over the total steps
            for i in tqdm.trange(self.total_steps, desc="Training Progress"):
                # Perform one training step
                loss_dict = self.step()




                # Log metrics periodically
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)




                # Save checkpoint periodically
                # Save at i+1 so the checkpoint name reflects the step *completed*
                if (i + 1) % self.cfg["save_every"] == 0:
                    self.save()




                # Optional: Add early stopping logic here if desired




        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        except Exception as e:
              print(f"\nAn error occurred during training: {e}")
              import traceback
              traceback.print_exc() # Print detailed traceback
        finally:
            # --- Final Save ---
            # Always save the final model state, regardless of how training ended
            print("\n--- Training Finished or Interrupted ---")
            self.save()
            if wandb.run is not None:
                wandb.finish() # Ensure WandB run is marked as finished
                print("WandB run finished.")