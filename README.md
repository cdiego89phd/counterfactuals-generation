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
Example script: python gpt2_cad_fine_tuning_trainer.py 
--setting_path /sentiment_task/fine_tuning_experiments/settings/
--setting_name example_cad_tuning.yaml
--wandb_key <key>
--wandb_project <project_name>
--debug_mode 0

    
### Generate counterfactuals
Example script: python generator.py 
--setting_path /sentiment_task/fine_tuning_experiments/settings/
--setting_name example_generation.yaml
--debug_mode 1

### Tune hyperparameters with wandb sweep
Firstly, you need to set a sweep in wandb with a proper hyperparameter space (see fine_tuning_experiments/settings/example_sweep_cad_tuning.yaml)
    
Example script: python wandb_gpt2_cad_fine_tuning_sweep.py --setting_path /sentiment_task/fine_tuning_experiments/settings/
--setting_name example_sweep_cad_tuning.yaml
--wandb_key <key>
--wandb_project <project_name>
--sweep_id <id>
--debug_mode 1
    
Similarly, wandb_gpt2_gen_tuning_sweep can be used (with similar params) to tune the generation hyperparameters with the wandb sweep.

### Evaluate generated counterfactuals
Takes in the generated counterfactuals and perform the evaluation (logging metrics on wandb)
    
Example script: python wandb_evaluate_counterfactuals.py --generation_path /sentiment_task/fine_tuning_experiments/generation/
--results_filename <filename>
--classifier_name distilbert-base-uncased-finetuned-sst-2-english
--n_counter_generated 3
--calculate_corr True
--wandb_key <key>
--wandb_project <project_name>
--lm_name <name>
--eval_task_name cad_fine_tuning
--metrics gbcd
