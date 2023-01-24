import transformers
import wandb
import datasets

import utils


def freeze_layers_lm(to_freeze, n_to_unfreeze, model):
    if to_freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False

        for i, m in enumerate(model.transformer.h):
            # Only un-freeze the last n transformer blocks
            if i+1 > len(model.transformer.h) - n_to_unfreeze:
                for parameter in m.parameters():
                    parameter.requires_grad = True

        for parameter in model.transformer.ln_f.parameters():
            parameter.requires_grad = True

        for parameter in model.lm_head.parameters():
            parameter.requires_grad = True
        print(f"Freezed the first {len(model.transformer.h)-n_to_unfreeze} model's layers")
        print(f"Only the last {n_to_unfreeze} model's layers will be trained!")
    else:
        print("All the model's layers will be trained!")

    return model


def prepare_training(df_trainset,
                     df_valset,
                     tokenizer,
                     tokenize_in_batch) -> (datasets.Dataset, datasets.Dataset):

    # convert dataset from pandas to Dataset
    training_set = datasets.Dataset.from_pandas(df_trainset)
    val_set = datasets.Dataset.from_pandas(df_valset)

    # TOKENIZE datasets
    tokenized_train = training_set.map(lambda examples: tokenizer(examples["wrapped_input"],
                                                                  padding="max_length",
                                                                  truncation=True), batched=tokenize_in_batch)
    tokenized_train = tokenized_train.add_column("labels", tokenized_train['input_ids'])
    tokenized_val = val_set.map(lambda examples: tokenizer(examples["wrapped_input"],
                                                           padding="max_length",
                                                           truncation=True), batched=tokenize_in_batch)
    tokenized_val = tokenized_val.add_column("labels", tokenized_val['input_ids'])
    return tokenized_train, tokenized_val


def train(out_dir, lm, trainset, valset, no_cuda, training_cfgs, project_name,
          run_name=None, save_model=True, is_sweep=False):

    with wandb.init(project=project_name, name=run_name):
        if is_sweep:
            # use wandb sweep config dict
            for k in wandb.config.keys():
                training_cfgs[k] = wandb.config[k]

        lm = freeze_layers_lm(training_cfgs['FREEZE_LAYERS'], training_cfgs['UNFREEZE_LAST_N'], lm)

        early_stopping = transformers.EarlyStoppingCallback(early_stopping_patience=training_cfgs['STOPPING_PATIENCE'])

        training_args = transformers.TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=True,
            no_cuda=no_cuda,
            num_train_epochs=training_cfgs['MAX_EPOCHS'],
            per_device_train_batch_size=training_cfgs['TRAIN_BATCHSIZE'],
            per_device_eval_batch_size=training_cfgs['EVAL_BATCHSIZE'],
            gradient_accumulation_steps=training_cfgs['BATCH_UPDATE'],
            do_eval=True,
            evaluation_strategy=transformers.IntervalStrategy.EPOCH,
            warmup_steps=training_cfgs['WARMUP_STEPS'],
            learning_rate=training_cfgs['LR'],
            adam_epsilon=training_cfgs['ADAM_EPS'],
            weight_decay=training_cfgs['WEIGHT_DECAY'],
            save_total_limit=1,
            save_strategy=transformers.IntervalStrategy.EPOCH,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            fp16=training_cfgs['fp16'],
            optim=training_cfgs['optim'],
        )

        trainer = transformers.Trainer(
            model=lm,
            args=training_args,
            train_dataset=trainset,
            eval_dataset=valset,
            callbacks=[early_stopping]
        )

        utils.print_gpu_utilization()
        trainer.train()

        if save_model:
            trainer.save_model()
