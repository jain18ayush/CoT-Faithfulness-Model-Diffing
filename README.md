# CoT-Faithfulness-Model-Diffing
Model Diffing a model organism of CoT unfaithfulness


# Steps
1. Create a dataset of unfaithful chain of thought; https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT
2. fine-tune a model : gpt-oss? -- whatever ,, use the SFT library maybe alongisde unsloth ids https://huggingface.co/docs/trl/en/unsloth_integration
3. measure the chain of thought faithfulness 
4. Set up a crosscoder on base and fine-tuneed
5. pick the 5 most changed latents 
6. apply that as a steering vector and measure faithfulness changed 

# Section 1 
[Chain of Thought Dataset from KAIST](https://github.com/kaistAI/CoT-Collection)
- Create a script to swap out chain of thoughts 


# Notes
- CoT-Train Formatted is the formatted data for fine-tuning 
- DeepSeek-R1-Distill-Qwen-7B    : This is the model we are using for fine-tuning 


# Learnings
- Create validation functions everywhere --> 

gsm8k100 (1).csv → 20% accuracy

gsm8k100_base.csv → 41% accuracy

> **⚠️ WRITE EVALS FIRST!!**
