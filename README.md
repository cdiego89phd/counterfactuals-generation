# counterfactuals-generation

This repositoty refers to the paper "CouRGe: Counterfactual Reviews Generator for Sentiment Analysis", under revision at AICS 22 (https://aics2022.mtu.ie/)

**ABSTRACT**: "Past literature in Natural Language Processing (NLP) has demonstrated that counterfactual data points are useful, for example, for increasing model generalisation, enhancing model interpretability, and as a data augmentation approach. However, obtaining counterfactual examples often requires human annotation effort, which is an expensive and highly skilled process. For these reasons, solutions that resort to transformer-based language models have been recently proposed to generate counterfactuals automatically, but such solutions show limitations.

In this paper, we present CouRGe, a language model that, given a movie review (i.e. a seed review) and its sentiment label, generates a counterfactual review that is close (similar) to the seed review, but of the opposite sentiment. CouRGe is trained by supervised fine-tuning GPT-2 on a task-specific dataset of paired movie reviews, and its generation is prompt-based. The model does not require any modification to the network's architecture nor the design of a specific new task for fine-tuning. 

Experiments show that CouRGe's generation is effective at flipping the seed sentiment and produces counterfactuals reasonably close to the seed review. This proves once again the great flexibility of language models towards downstream tasks as hard as counterfactual reasoning and opens up the use of CouRGe's generated counterfactuals for the applications mentioned above."

## Project's structure (Python 3.7)

- the root contains scripts for fine-tuning a language model (with a specific dataset such as Rotten Tomatoes used in this paper) 
- **sentimen_task** contains the resources to generate counterfacuals for the Sentiment Analysis task
    - **notebooks** stores notebooks to prepare the datasets for the experiments  
    - **cad_imdb**, **imdb_pang**, **yelp** are folders that store different sentiment analysis datasets;
    - **fine_tuning_experiments** runs and store results for fine-tuning GPT2 with the cad_imdb dataset;
    - **zero_shot_experiments** store results for the zero-shot generation;
    - **ood_experiments** runs and store results of the Out-Of-Distribution experiments on yelp and imdb_pang;
    - *generator.py* runs the counterfactual generation;
    - *wandb_evaluate_counterfactuals.py* runs the evaluation for the generated counterfactuals, logging results on wandb;
    - *evaluate_counters_from_local.py* runs the evaluation for the generated counterfactuals, displaying the results locally;
 
Most of the code is also integrated with the Weights & Biases logging system.

### Fine-tune the generator (on GPT2)
Example script: python gpt2_cad_fine_tuning_trainer.py TODO

### Generate counterfactuals
Example script: python generator.py TODO

### Tune hyperparameters with wandb sweep
Example script: python wandb_gpt2_cad_fine_tuning_sweep.py TODO

### Evaluate generated counterfactuals
Example script: python wandb_evaluate_counterfactuals.py TODO
