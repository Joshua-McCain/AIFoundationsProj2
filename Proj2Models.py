import torch
import torch.nn as nn
from abc import ABC, abstractmethod

#Abstract class so a base model class with no forward cannot be instantiated
class BaseLanguageModel(nn.Module, ABC):

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=6, dropout=0.2, pad_token_id=0, top_p=.8, name="Language Model"):
        """
        This generalized __init__() function serves to standardize hyperparameter settings in our language models
        See the following parameters for details:

        :param vocab_size: Vocab size of the tokenizer used. For this project, it was 10,000.
        :param embed_dim: Embedding dimension used to map tokens to embed_dim length vectors.
        :param hidden_dim: Hidden dimension mapping out of the model specific layer.
        :param num_layers: The number of layers of the specific model being implemented. For example, RNN would have num_layers layers.
        :param dropout: Layer dropout rate used to reduce reliance of specific pathways in node structures by zeroing out nodes.
        :param pad_token_id: Padding ID used by the tokenizer. Ensures the embedding layer has a 0 vector for that index.
        :param top_p: Range [0, 1] Nucleus sampling for predicting tokens threshold. 0 is including the most common token, 1 is all tokens .
        :param name: The name which the prompts and graph refer to this model as.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.top_p = top_p
        self.name = name

    @abstractmethod
    def forward(self, input_ids, hidden=None):
        """ 
        Passes the input prompt through the network defined in children, eventually computing the logits

        NOTE: The logits, when returned, are of dimension (batch_size, seq_len, vocab_size), where
        batch_size (B): Number of batches for this call
        seq_len (S): The length of the prompts, with each corresponding to a token in its sequence (likely padded with <pad> tokens)
        vocab_size (V): A probability distribution of word the model predicts should be in that position
        
        So, the logits form a 3D tensor, where each prediction is held on the third dimension. This is important for
        computing the loss later.

        :param input_ids: The batched input prompts, already tokenized
        :param hidden: Potentially the hidden data of the previous run
        :return: The output logits of shape (B x S x V) and potentially extra information such as the hidden information of this forward
        """
        pass

    def predict_next_token(self, input_ids, temperature=1.0):
        """ 
        Used iteratively in the prompt() function, this function takes the current prompt as input_ids and,
        augmenting the logits with the temperature value, gives a prediction of the next token that can be
        found in the given sequence.

        :param input_ids: The tokens that are used to predict the next token in that sequence.
        :param temperature: a positive float that determines model creativity, 0 is uncreative (do not give it 0, div by 0 error), 1 is no change, > 1 more creative
        :return: The predicted next token in the provided sequence.
        """

        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(input_ids)
            # Temperature scaling for more creative/static token generation
            # Apply temperature to the LAST TOKEN prob distribution, this allows us to predict the next token of the completion
            logits = logits[:, -1, :]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=1)

            # Instead of choosing the best option, we want to implement top-p sampling for unique outputs
            # Obtain the descending order cumulative distribution of the tokens along the probability dimension
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
            # Create a mask of the tokens we want to remove probability for such that lower prob tokens are discarded
            sorted_mask = cumulative_probs > self.top_p

            """
            Before eliminating probabilities, we may have to account for the case where sorted_mask has all True values.
            In other words, all tokens could be eliminated if the first probability is so likely that it exceeds argument top_p.
            
            To prevent this, we make sure the first token is always false in the sorted_mask by
            shifting the entire tensor along the last dimension (where probs will be) one element to the right.
            This also means the element that took us over the top-p threshold is included.
            """
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            
            # Eliminate usused probabilities
            sorted_probs[sorted_mask] = 0
            # Normalize the new probabilities since we removed some
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
            # Using the new probabilities, we can sample from them to obtain our top-p token
            next_token_index = torch.multinomial(sorted_probs, num_samples=1)
            # Finally, get the next token id through a generic multi-dimensional method, accessing the last dimension for the next token index
            # For our 1D tensor, using .item() on this will give the next generated token.
            next_token_id = sorted_indices.gather(-1, next_token_index)

            return next_token_id.item(), hidden
    
    def prompt(self, tokenizer, prompt, max_seq_length=50, eos_token_id=None, temperature=1.0, device='cpu'):
        """ 
        Generates a response, token by token, to the prompt given to the model.

        :param tokenizer: The tokenizer that was used to structure the data for training.
        :param prompt: The input string used to generate a response.
        :param max_seq_length: The max length the response can be, in tokens. Prevents rambling responses from generating infinitely.
        :param eos_token_id: If the eos token is generated, generation ends.
        :param temperature: Determines how creative responses are. See predict_next_token() comment.
        :param device: Device the model should use for generation. Use "cuda" if you have it.
        """

        self.eval()
        # Convert the prompt to tokenized sequence for model comprehension
        input_ids = tokenizer.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = [] # Storing response

        # Responses can be a maximum of max_length tokens long
        for _ in range(max_seq_length):
            # Get the next token based on our prompt
            next_token_id, hidden = self.predict_next_token(input_tensor, temperature)
            # Check if we are using eos token ID, then ask if what was generated was that ID so we can end early
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
            # Build our response with next token
            generated_ids.append(next_token_id)
            # Now that we have the next token, remake the input to include only the new token for further generation.
            # This enables the autoregressive (Causal) generation of the prompt.
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        # Once we have an eos token or the max length is reached, return the decoded prompt
        return tokenizer.decode(generated_ids, out_type=str)
    

class RNNLanguageModel(BaseLanguageModel):
    def __init__(self, *args, **kwargs):
        """ 
        The RNN implementation of the base class. See the BaseLanguageModel for in depth hyperparameter explanation
        """
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_token_id)
        self.rnn = nn.RNN(self.embed_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)


    def forward(self, input_ids, hidden=None):
        """ 
        The RNN implementation of layer propagation. See the BaseLanguageModel for in depth hyperparameter and output explanation.
        The fully connected layer at the end gives us our probability distributions.
        """


        embeds = self.embedding(input_ids)
        out, hidden = self.rnn(embeds, hidden)
        logits = self.fc(out)
        return logits, hidden
    

class LSTMLanguageModel(BaseLanguageModel):
    def __init__(self, *args, **kwargs):
        """ 
        The LSTM implementation of the base class. See the BaseLanguageModel for in depth hyperparameter explanation
        """
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_token_id)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)


    def forward(self, input_ids, hidden=None):
        """ 
        The LSTM implementation of layer propagation. See the BaseLanguageModel for in depth hyperparameter and output explanation.
        The fully connected layer at the end gives us our probability distributions.
        """


        embeds = self.embedding(input_ids)
        out, hidden = self.lstm(embeds, hidden)
        logits = self.fc(out)
        return logits, hidden


class TransformerLanguageModel(BaseLanguageModel):
    def __init__(self, n_heads, *args, **kwargs):
        """ 
        The Transformer implementation of the base class. See the BaseLanguageModel for in depth hyperparameter explanation.
        We have 1 additional argument n_heads

        :param n_heads: The number of attention heads used for the Transformer block
        """
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_token_id)
        self.positional_embed = PositionalEncoding(self.embed_dim, dropout=self.dropout)

        self.encode_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                       nhead=n_heads,
                                                       dim_feedforward=self.hidden_dim,
                                                       dropout=self.dropout,
                                                       batch_first=True
                                                      )
        self.transformer = nn.TransformerEncoder(self.encode_layer, num_layers=self.num_layers)
        
        self.fc = nn.Linear(self.embed_dim, self.vocab_size)


    def forward(self, input_ids, hidden=None):
        """ 
        The Transformer implementation of layer propagation. See the BaseLanguageModel for in depth hyperparameter and output explanation.
        The fully connected layer at the end gives us our probability distributions.
        Additionally, note the use of the mask and positional embedding modifications so generation of sentences with the Transformer functioned
        """

        embeds = self.embedding(input_ids)
        #Note the positional_embed layer automatically adds the embeds so you dont have to add them yourself.
        embeds = self.positional_embed(embeds)

        # Creates an upper triangular matrix full of negative infinity. This stops the model from looking ahead in the sequence and
        # "cheating," seeing the correct token and predicting that token every time. This leads to immediate overfitting and causes
        # prompt completion to repeat the last token it saw for every iteration.
        # This is what makes the model a decoder model used for generation.
        seq_len = input_ids.size(1)
        mask = generate_square_subsequent_mask(seq_len, device=input_ids.device)

        out = self.transformer(embeds, mask=mask)
        logits = self.fc(out)
        return logits, hidden


#Extra layer provided by Dr. James Ghawaly for the positional embedding needed for transformer models to understand token order
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Upper triangular negative infinity mask generation
def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask