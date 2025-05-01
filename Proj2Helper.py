import os
import json
import torch
from torch.utils.data import Dataset

# For tokenizer training
def merge_text_files(directory, outfile_name):
    """
    This function takes all of the raw data and uses it to train the bpe tokenizer such that,
    when we have our eventual prompts and answers, we can easily convert them into tokenized sequences
    for training our models and decode them for reading. BPE tokenization allows for the token sequences to be condensed
    to small sequences of letters or even full words for easier processing and better pattern recognition.

    :param directory: The folder containing all of the raw data.
    :param outfile_name: The corpus file name to write to
    """
    # We want a clean file when we call this function
    if os.path.exists(outfile_name):
        os.remove(outfile_name)

    # We will merge all of the text files in the specified directory into one file so it can be used for training the tokenizer
    with open(outfile_name, 'w', encoding='utf-8') as out:
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as cur_data:
                    out.write(cur_data.read())
                    out.write('\n')

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_seq_len=128):
        """ 
        Takes the directory of a jsonl structured with prompt and completion pairs, encodes them, and adds them to the samples list

        :param filepath: Location of the .jsonl file with prompts and completions
        :param tokenizer: Tokenizer we want to use for the training
        :param max_seq_len: A limit for how long we will allow a prompt to be from the .jsonl
        """

        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # With the provided jsonl files from the handout...
        with open(filepath, "r", encoding="utf-8") as file:
            # Line by line (with each line being a prompt/completion pair)...
            for line in file:
                item = json.loads(line)
                # Take the whole line and put it into one string variable
                text = item["prompt"] + " " + item["completion"]
                # Tokenize the line, making sure the line is not too long
                tokens = tokenizer.encode(text, out_type=int)[:max_seq_len]
                # We also don't want lines that are too short and don't provide useful info
                if len(tokens) < 2:
                    continue
                # Now that our jsonl line is tokenized, record the result
                # Note that is is a JAGGED 2D array, meaning we will need to add padding later
                self.samples.append(tokens)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """ 
        Takes the samples from initialization and modifies them for teacher forcing for the models

        :param index: The sample being references
        :return: The inputs and target for that sample
        """

        tokens = self.samples[index]
        # Creating the inputs and labels such that we can train by teacher forcing.
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids
    
