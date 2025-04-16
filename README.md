This repository is the code for Project 2 of Dr. James Ghawaly's Foundational Models course.

The main code to run the project can be found in the Jupyter Notebook Project2.ipynb. Please load this Python3 project with all of the required dependencies. The code was developed with the following dependencies and versions:

PyTorch 2.6.0
SentencePiece (BPE tokenizer) 0.2.0 (found here https://github.com/google/sentencepiece)
Matplotlib 3.10.0
nltk 3.9.1


TO RUN THE CODE:
TLDR: For running the models, select run all code blocks once the constants at the top are modified to your liking, and follow the prompts the code gives you for input.

The modifiable constants that control all of the models' hyperparameters at once can be found at the very top of the notebook under the "Constants" secondary header. Once the constants are satisfactory, you can preload all of the necessary data and functions by running all of the code blocks under the top header "Function/Class Definitions & Setup." Collapse this header and the remaining code is found under the "Main Code" header.

Please run the "Tokenizer & Preliminary Vars" (type y if you need to train the tokenizer for the .model file) before training the models. From here, run any code block for training a model and subsequently run the prompt block if you want to prompt the trained model.