from tkinter import *
import tkinter.ttk as ttk
import yaml
import utils
import generator
# from tkinter import ttk

MODELS_PATH = "/home/diego/counterfactuals-generation/sentiment_task/fine_tuning_experiments/saved_models"
LM_OPTIONS = ["gpt2-medium",
              "LOC-sshleifer/tiny-gpt2@prompt-1@fold-0@cad_fine_tuning"]
TOKENIZER_OPTIONS = ["gpt2-medium",
                     "gpt2-small"]
PROMPT_OPTIONS = ["1", "2"]
GEN_FILE_OPTIONS = ["console_generation.yaml"]
CLASS_OPTIONS = ["Positive", "Negative"]
SPECIAL_TOKENS = {
    "bos_token": "<|BOS|>",
    "eos_token": "<|EOS|>",
    "unk_token": "<|UNK|>",
    "pad_token": "<|PAD|>",
    "sep_token": "<|SEP|>"
}
# TODO fix prompt 2
MAP_PROMPT = {"1": "<bos_token><label_ex> review:<sep><example_text><sep><label_counter> review:<sep>",
              "2": "<bos_token>The movie is <label_ex>. <example_text> The movie is <label_counter>."}


def update_status(label, new_text):
    label.config(text=new_text)


def trigger_generation(window,
                       model,
                       base_tokenizer,
                       prompt,
                       gen_yaml,
                       seed_text,
                       review_class,
                       counter_box,
                       status_label):

    # extract parameters from forms
    update_status(status_label, "Generation in progress...")
    window.update()

    # load the language model
    model_name = model.get()
    if model_name.split("-")[0] == "LOC":
        # load model from local
        name = model_name.replace("LOC-", "")
        lm = utils.load_gpt2_from_local(f"{MODELS_PATH}/{name}")
        tokenizer, _, _ = utils.load_gpt2_objects(base_tokenizer, SPECIAL_TOKENS)
        print("Language model loaded from local directory")
    else:
        # load from huggingface
        tokenizer, lm, _ = utils.load_gpt2_objects(model_name, SPECIAL_TOKENS)
        print("Language model loaded from huggingface")

    prompt = MAP_PROMPT[prompt.get()]
    yaml_file = open(gen_yaml.get())
    parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)
    seed_review = seed_text.get(1.0, "end")
    seed_class = review_class.get()

    # generate counterfactual
    # TODO complete
    counter_text = generator.generate_single_counterfactual(parsed_yaml_file,
                                                            prompt,
                                                            seed_review,
                                                            seed_class,
                                                            lm,
                                                            tokenizer)

    # report the generation to the form
    counter_box.delete(1.0, "end")  # clear the text box
    counter_box.insert(1.0, counter_text)  # set new text

    print("Generation COMPLETED!")
    update_status(status_label, "Generation COMPLETED!")


def main():
    window = Tk()
    window.title("Console for Counterfactuals Generation")
    window.geometry('600x600')
    window.configure(background="grey")

    # TODO create a "set to default forms button"

    # 1. Select language model (drop down menu)
    model_label = ttk.Label(window, text="Select language model")
    model_label.pack()
    model_clicked = StringVar()
    model_menu = ttk.OptionMenu(window, model_clicked, LM_OPTIONS[0], *LM_OPTIONS)
    model_menu.pack()

    # 2. Select base tokenizer (drop down menu)
    token_label = ttk.Label(window, text="Select base tokenizer")
    token_label.pack()
    tokenizer_clicked = StringVar()
    token_menu = ttk.OptionMenu(window, tokenizer_clicked, TOKENIZER_OPTIONS[0], *TOKENIZER_OPTIONS)
    token_menu.pack()

    # 3. Select prompt (drop down menu)
    prompt_label = ttk.Label(window, text="Select prompt")
    prompt_label.pack()
    prompt_clicked = StringVar()
    prompt_menu = ttk.OptionMenu(window, prompt_clicked, PROMPT_OPTIONS[0], *PROMPT_OPTIONS)
    prompt_menu.pack()

    # 4. Select generation parameters (drop down menu)
    gen_label = ttk.Label(window, text="Select generation parameters")
    gen_label.pack()
    gen_clicked = StringVar()
    gen_menu = ttk.OptionMenu(window, gen_clicked, GEN_FILE_OPTIONS[0], *GEN_FILE_OPTIONS)
    gen_menu.pack()

    # 5. Write seed review text (text box)
    seed_text_box = Text(
        window,
        height=6,
        width=50
    )
    seed_text_box.pack(expand=True)
    seed_text_box.insert('end', "Insert here the seed review")
    seed_text_box.config(state='normal')

    # 6. Select seed review class (drop down menu)
    class_label = ttk.Label(window, text="Select seed review class")
    class_label.pack()
    class_clicked = StringVar()
    class_menu = ttk.OptionMenu(window, class_clicked, CLASS_OPTIONS[0], *CLASS_OPTIONS)
    class_menu.pack()

    # 7. Status of generation
    status_label = ttk.Label(window, text="Push the button to trigger the generation")
    status_label.pack()

    # 8. Show the counterfactual review
    counter_text_box = Text(
        window,
        height=6,
        width=50
    )
    counter_text_box.insert('end', "Counterfactual review will be shown here")
    counter_text_box.config(state='normal')

    # 7. Button to trigger generation
    generation_button = ttk.Button(window, text="Generate!",
                                   command=lambda: trigger_generation(window,
                                                                      model_clicked,
                                                                      tokenizer_clicked,
                                                                      prompt_clicked,
                                                                      gen_clicked,
                                                                      seed_text_box,
                                                                      class_clicked,
                                                                      counter_text_box,
                                                                      status_label))

    generation_button.pack()
    counter_text_box.pack(expand=True)

    window.mainloop()


if __name__ == "__main__":
    main()

# b = Label(window ,text = "Last Name").grid(row = 1,column = 0)
# c = Label(window ,text = "Email Id").grid(row = 2,column = 0)
# d = Label(window ,text = "Contact Number").grid(row = 3,column = 0)
# Entry(window).grid(row = 0,column = 1)
# b1 = Entry(window).grid(row = 1,column = 1)
# c1 = Entry(window).grid(row = 2,column = 1)
# d1 = Entry(window).grid(row = 3,column = 1)


# def clicked():
#     res = "Welcome to " + txt.get()
#     lbl.configure(text=res)
#
# btn = ttk.Button(window ,text="Submit").grid(row=4,column=0)

# def save_selected_values():
#     global values1
#     values1 = [mymenu1.var.get(), mymenu2.var.get(), mymenu3.var.get(), mymenu4.var.get(), ent1.get(), mymenu5.var.get()]
#     print(values1)