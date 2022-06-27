import transformers
import wandb
import datasets


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


# TODO is this method general or specific to gpt2? Let's see
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


def train(out_dir, lm, trainset, valset, no_cuda, training_cfgs, project_name, run_name=None, save_model=True):

    if training_cfgs is None:
        # use wandb sweep config dict
        training_cfgs = wandb.config

    max_epochs = training_cfgs['MAX_EPOCHS']
    train_batch_size = training_cfgs['TRAIN_BATCHSIZE']
    eval_batch_size = training_cfgs['EVAL_BATCHSIZE']
    batch_update = training_cfgs['BATCH_UPDATE']
    warmup_steps = training_cfgs['WARMUP_STEPS']
    lr = training_cfgs['LR']
    adam_epsilon = training_cfgs['ADAM_EPS']
    weight_decay = training_cfgs['WEIGHT_DECAY']
    stopping_patience = training_cfgs['STOPPING_PATIENCE']
    to_freeze_layers = training_cfgs['FREEZE_LAYERS']
    unfreeze_last_n = training_cfgs['UNFREEZE_LAST_N']

    lm = freeze_layers_lm(to_freeze_layers, unfreeze_last_n, lm)

    early_stopping = transformers.EarlyStoppingCallback(early_stopping_patience=stopping_patience)
    with wandb.init(project=project_name, name=run_name):
        training_args = transformers.TrainingArguments(
            output_dir=out_dir,
            no_cuda=no_cuda,
            num_train_epochs=max_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=batch_update,
            do_eval=True,
            evaluation_strategy=transformers.IntervalStrategy.EPOCH,
            warmup_steps=warmup_steps,
            learning_rate=lr,
            adam_epsilon=adam_epsilon,
            weight_decay=weight_decay,
            save_total_limit=1,
            save_strategy=transformers.IntervalStrategy.EPOCH,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss'
        )

        trainer = transformers.Trainer(
            model=lm,
            args=training_args,
            train_dataset=trainset,
            eval_dataset=valset,
            callbacks=[early_stopping]
        )

        trainer.train()

        if save_model:
            trainer.save_model()
