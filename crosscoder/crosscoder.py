import torch
from datasets import load_dataset
import os
from pathlib import Path
import argparse
import json

# Define available data types
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


# Define a structure for the loss output for clarity
class LossOutput(NamedTuple):
    l2_loss: torch.Tensor           # Reconstruction error
    l1_loss: torch.Tensor           # Sparsity penalty
    l0_loss: torch.Tensor           # Count of active features
    explained_variance: torch.Tensor # Overall variance explained
    explained_variance_A: torch.Tensor # Variance explained for Model A
    explained_variance_B: torch.Tensor # Variance explained for Model B


class CrossCoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = self.cfg["dict_size"]
        self.device = self.cfg["device"] # Store the device as an instance attribute
        d_in = self.cfg["d_in"]
        n_models = 2
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        torch.manual_seed(self.cfg["seed"])


        self.W_enc = nn.Parameter(torch.empty(n_models, d_in, d_hidden, dtype=self.dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, n_models, d_in, dtype=self.dtype))
        self.b_dec = nn.Parameter(torch.zeros((n_models, d_in), dtype=self.dtype))


        nn.init.normal_(self.W_dec, std=1.0)
        dec_norm = self.W_dec.norm(dim=-1, keepdim=True)
        self.W_dec.data /= (dec_norm + 1e-8)
        self.W_dec.data *= self.cfg["dec_init_norm"]
        self.W_enc.data = einops.rearrange(self.W_dec.data.clone(), "h n d -> n d h")


        self.d_hidden = d_hidden
        self.to(self.cfg["device"])
        self.save_dir = None
        self.save_version = 0
        print(f"CrossCoder (BatchTopK variant) initialized on device: {self.cfg['device']}")
        if self.cfg.get("sparsity_type", "l1") == "batch_top_k":
              assert "k_sparsity" in self.cfg, "k_sparsity must be in cfg for BatchTopK"
              print(f"Using BatchTopK sparsity with k={self.cfg['k_sparsity']}")




    def get_pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates pre-activations (before ReLU or TopK)."""
        x_enc = einops.einsum(x, self.W_enc, "b n d, n d h -> b h")
        pre_acts = x_enc + self.b_enc # Shape: [batch, d_hidden]
        return pre_acts


    def encode(self, x: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        """
        Encodes input activations. Applies BatchTopK if training and configured.
        Otherwise, applies ReLU (for inference or L1 mode).
        """
        pre_acts = self.get_pre_activations(x) # Shape: [batch, d_hidden]


        if is_training and self.cfg.get("sparsity_type", "l1") == "batch_top_k":
            # BatchTopK logic (as described in the paper for f_train)
            # 1. Calculate scaling values v(xi, j) = fj(xi) * (||d_base_j||_2 + ||d_chat_j||_2)
            #    where fj(xi) is the ReLU of pre_acts.
            #    For simplicity, we'll use pre_acts directly for ranking initially,
            #    as the paper's v calculation is based on fj(xi) (post-ReLU)
            #    and decoder norms, which can be computationally intensive per step.
            #    A common simplification in SAEs is to use pre-ReLU for TopK.
            #    Let's refine this based on the paper's f_train definition:
            #    v(xi,j) = ReLU(pre_acts[i,j]) * (||W_dec[j,0,:]|| + ||W_dec[j,1,:]||)


            relu_acts = F.relu(pre_acts) # fj(xi)


            with torch.no_grad(): # Decoder norms don't need gradients here
                decoder_norms_sum = self.W_dec.norm(p=2, dim=-1).sum(dim=-1) # Shape [d_hidden]
                                                                        # (||d_base_j|| + ||d_chat_j||)


            v_values = relu_acts * decoder_norms_sum.unsqueeze(0) # Shape [batch, d_hidden]


            k = self.cfg["k_sparsity"]
            num_inputs = x.shape[0] # batch_size
            num_to_select_total = num_inputs * k # Total top activations across the batch


            # Flatten v_values to find global top-k across batch and features
            top_k_values, _ = torch.topk(v_values.flatten(), k=num_to_select_total)


            if top_k_values.numel() == 0: # Handle case where no values are selected (e.g. k=0)
                  # If k=0 or no positive activations, all are zero
                  threshold = float('inf')
            elif num_to_select_total == 0: # k=0
                  threshold = float('inf')
            else:
                  # The threshold is the k-th largest value
                  threshold = top_k_values[-1]




            # Create a mask for activations >= threshold
            # Only keep activations if their v_value was among the top k*N
            train_acts_mask = v_values >= threshold


            # Apply the mask to the ReLU activations (not pre_acts directly for f_train)
            # This ensures f_train (output here) matches the paper's f_j(x_i) for selected, 0 otherwise
            final_acts = relu_acts * train_acts_mask.float()


        else: # Standard ReLU for inference or if L1 sparsity is used
            final_acts = F.relu(pre_acts)


        return final_acts




    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        # (decode function remains the same)
        acts_dec = einops.einsum(acts, self.W_dec, "b h, h n d -> b n d")
        x_reconstruct = acts_dec + self.b_dec
        return x_reconstruct


    def forward(self, x: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        # Pass is_training to encode
        acts = self.encode(x, is_training=is_training)
        x_reconstruct = self.decode(acts)
        return x_reconstruct


    def get_losses(self, x: torch.Tensor, is_training: bool = True) -> LossOutput:
        x = x.to(self.cfg["device"]).to(self.dtype)


        # --- Forward pass (use is_training for encode) ---
        acts = self.encode(x, is_training=is_training)
        x_reconstruct = self.decode(acts)


        # --- L2 Loss (Mean Squared Error) ---
        diff = (x_reconstruct - x).float()
        squared_diff = diff.pow(2)
        l2_per_batch = einops.reduce(squared_diff, 'b n d -> b', 'sum')
        l2_loss = l2_per_batch.mean()


        # --- Sparsity Related Losses ---
        l1_loss_val = torch.tensor(0.0, device=self.device) # Default for BatchTopK
        l0_loss_val = (acts > 1e-8).float().sum(dim=-1).mean() # L0 is always informative


        if self.cfg.get("sparsity_type", "l1") == "l1":
            # Calculate L1 loss if configured (original method)
            with torch.no_grad():
                  decoder_norms = self.W_dec.norm(dim=-1)
                  total_decoder_norm = einops.reduce(decoder_norms, 'h n -> h', 'sum')
            l1_loss_val = (acts.float() * total_decoder_norm[None, :]).sum(dim=-1).mean()
        elif self.cfg.get("sparsity_type", "l1") == "batch_top_k":
            # For BatchTopK, the primary loss is L2.
            # The paper mentions an auxiliary loss (L_aux) for recycling dead latents.
            # We'll add a placeholder; a simple L_aux could be an L1 on pre-activations
            # or encouraging dead neurons to fire. This requires more careful implementation
            # from the Bussmann et al. paper or a similar strategy.
            # For now, we only have the L2 and the L0 is just for monitoring.
            # The sparsity is *enforced* by the TopK selection in encode().
            pass # No direct L1 term. Sparsity is structural.


        # --- Explained Variance ---
        with torch.no_grad():
            variance = einops.reduce((x - x.mean(dim=0, keepdim=True)).pow(2), 'b n d -> b', 'sum')
            explained_variance = (1 - l2_per_batch / (variance + 1e-8)).mean()
            variance_A = (x[:, 0] - x[:, 0].mean(dim=0, keepdim=True)).pow(2).sum(dim=-1)
            l2_per_batch_A = squared_diff[:, 0].sum(dim=-1)
            explained_variance_A = (1 - l2_per_batch_A / (variance_A + 1e-8)).mean()
            variance_B = (x[:, 1] - x[:, 1].mean(dim=0, keepdim=True)).pow(2).sum(dim=-1)
            l2_per_batch_B = squared_diff[:, 1].sum(dim=-1)
            explained_variance_B = (1 - l2_per_batch_B / (variance_B + 1e-8)).mean()


        return LossOutput(l2_loss, l1_loss_val, l0_loss_val, explained_variance, explained_variance_A, explained_variance_B)


    def create_save_dir(self, base_dir_str="./checkpoints"):
        base_dir = Path(base_dir_str)
        base_dir.mkdir(parents=True, exist_ok=True) # Ensure base directory exists




        # Find existing version directories
        version_list = []
        for file in base_dir.iterdir():
            if file.is_dir() and file.name.startswith("version_"):
                try:
                    version_list.append(int(file.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue # Ignore directories not matching the pattern




        # Determine the next version number
        if version_list:
            version = 1 + max(version_list)
        else:
            version = 0




        self.save_dir = base_dir / f"version_{version}"
        self.save_dir.mkdir(parents=True)
        print(f"Created checkpoint directory: {self.save_dir}")


    def save(self, checkpoint_dir_str="./checkpoints"):
        """Saves the model state dictionary and config."""
        if self.save_dir is None:
            self.create_save_dir(checkpoint_dir_str)




        # Define file paths within the versioned directory
        weight_path = self.save_dir / f"crosscoder_{self.save_version}.pt"
        cfg_path = self.save_dir / f"crosscoder_{self.save_version}_cfg.json"




        # Save the model's learned parameters
        torch.save(self.state_dict(), weight_path)




        # Save the configuration used for this model
        with open(cfg_path, "w") as f:
            # Convert Path objects in config to strings for JSON serialization
            serializable_cfg = {k: str(v) if isinstance(v, Path) else v for k, v in self.cfg.items()}
            json.dump(serializable_cfg, f, indent=2)




        print(f"Saved checkpoint {self.save_version} to {self.save_dir}")
        self.save_version += 1 # Increment version for the next save


    @classmethod
    def load(cls, version_dir_str: str, checkpoint_version: int = 0):
          """Loads a CrossCoder model from a saved checkpoint directory."""
          save_dir = Path(version_dir_str)
          cfg_path = save_dir / f"crosscoder_{checkpoint_version}_cfg.json"
          weight_path = save_dir / f"crosscoder_{checkpoint_version}.pt"




          if not cfg_path.exists() or not weight_path.exists():
                raise FileNotFoundError(f"Checkpoint files not found in {save_dir} for version {checkpoint_version}")


          print(f"Loading config from: {cfg_path}")
          with open(cfg_path, "r") as f:
              cfg = json.load(f)
          print("Loaded Config:")
          pprint.pprint(cfg)




          # Create a new instance with the loaded config
          # Ensure device is handled correctly if loading on different hardware
          if 'device' not in cfg:
                cfg['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                print(f"Warning: 'device' not found in config, defaulting to {cfg['device']}")
          instance = cls(cfg=cfg)




          print(f"Loading weights from: {weight_path}")
          # Load the saved weights onto the correct device specified in the config
          state_dict = torch.load(weight_path, map_location=cfg["device"])
          instance.load_state_dict(state_dict)
          print("Weights loaded successfully.")




          instance.save_dir = save_dir # Set save_dir for potential future saves
          instance.save_version = checkpoint_version + 1 # Start saving from next version
          return instance


    @classmethod
    def load_from_hf(
            cls,
            repo_id: str = "ckkissane/crosscoder-gemma-2-2b-model-diff",
            path_in_repo: str = "blocks.14.hook_resid_pre", # Original path structure
            cfg_filename: str = "cfg.json",
            weights_filename: str = "cc_weights.pt",
            device: Optional[Union[str, torch.device]] = None,
            **kwargs # Pass extra args to hf_hub_download (e.g., token)
        ):
            print(f"Loading CrossCoder from HF Hub: {repo_id}/{path_in_repo}")


            # Construct full paths within the repo
            config_repo_path = f"{path_in_repo}/{cfg_filename}"
            weights_repo_path = f"{path_in_repo}/{weights_filename}"




            # Download config and weights files
            try:
                config_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=config_repo_path,
                    **kwargs
                )
                weights_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=weights_repo_path,
                    **kwargs
                )
            except Exception as e:
                print(f"Error downloading from Hugging Face Hub: {e}")
                raise
            print(f"Downloaded config: {config_path}")
            print(f"Downloaded weights: {weights_path}")


            # Load config from the downloaded JSON file
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            print("Loaded Config:")
            pprint.pprint(cfg)


            # Determine the device
            if device is None:
                # Use device from config if available, otherwise default
                if 'device' in cfg:
                    device = cfg['device']
                else:
                    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                    print(f"Warning: 'device' not found in config, defaulting to {device}")
            # Update config with the determined device (important for model initialization)
            cfg["device"] = str(device)
            # Initialize CrossCoder instance with the loaded config
            instance = cls(cfg)


            # Load weights onto the specified device
            state_dict = torch.load(weights_path, map_location=device)
            instance.load_state_dict(state_dict)
            print(f"Weights loaded successfully onto device: {device}")
            return instance