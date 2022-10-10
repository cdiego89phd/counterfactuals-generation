import transformers
import pandas as pd


def load_dataset(loading_path):
    train = pd.read_csv(loading_path + "training_set", sep='\t')
    val = pd.read_csv(loading_path + "val_set", sep='\t')
    test = pd.read_csv(loading_path + "test_set", sep='\t')
    return train, val, test


def wrap_dataset_with_prompt(df_row, template, mapping_labels, spec_tokens):
    final_text = template.replace("<label_ex>", mapping_labels[df_row["label_ex"]])
    final_text = final_text.replace("<example_text>", df_row["example"])
    final_text = final_text.replace("<label_counter>", mapping_labels[df_row["label_counter"]])
    if "<counter_text>" in final_text:
        final_text = final_text.replace("<counter_text>", df_row["counterfactual"])

    if spec_tokens != "None":
        final_text = final_text.replace("<sep>", spec_tokens["sep_token"])
        final_text = final_text.replace("<bos_token>", spec_tokens["bos_token"])
        final_text = final_text.replace("<eos_token>", spec_tokens["eos_token"])

    # this is for vanilla generation
    # we create a prompt with some words from the seed review
    words = df_row["example"].split(" ")
    final_text = final_text.replace("<0>", words[0])
    final_text = final_text.replace("<1>", words[1])
    final_text = final_text.replace("<2>", words[2])

    return final_text


def load_gpt2_objects(model_name, spec_tokens):
    # Load language gpt2 objects
    tok = transformers.GPT2Tokenizer.from_pretrained(model_name)
    print("Downloaded tokenizer!")
    if spec_tokens != "None":
        print(f"Len of tokenizer before adding tokens:{len(tok)}")
        tok.add_special_tokens(spec_tokens)  # add special tokens
        print("Added special tokens to tokenizer!")
        print(f"Len of tokenizer after adding tokens:{len(tok)}")

    model_config_class = transformers.GPT2Config.from_pretrained(model_name)
    model = transformers.GPT2LMHeadModel.from_pretrained(model_name, config=model_config_class)
    print("Downloaded model and cfg!")
    if spec_tokens != "None":
        # special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tok))

    return tok, model, model_config_class


def load_gpt2_from_local(model_path):
    model = transformers.GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
    return model


def prepare_classifier(classifier_name):
    # load the sentiment classifier
    classifier_tokenizer = transformers.AutoTokenizer.from_pretrained(classifier_name)
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(classifier_name)

    # TODO in the future, check this dict is the one used by the classifier
    # there will be different dicts for different classifiers
    classifier_label_map = {'NEGATIVE':  0, 'POSITIVE': 1}

    classification_tools = {"tokenizer": classifier_tokenizer,
                            "classifier": classifier,
                            "label_map": classifier_label_map}

    return classification_tools
