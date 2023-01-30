import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.legacy.datasets import Multi30k, WMT14
from torchtext.legacy.data import Field, BucketIterator
import spacy
import numpy as np
import random
from tqdm import tqdm
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_datasets(batch_size=128):
    # Download the language files
    # spacy_de = spacy.load('de_core_news_sm')k
    # spacy_en = spacy.load('en_core_news_sm')

    # # define the tokenizer
    # def tokenize_de(text):
    #     return [token.text for token in spacy_de.tokenizer(text)][::-1]

    # def tokenize_en(text):
    #     return [token.text for token in spacy_en.tokenizer(text)]

    # Create the pytext's Field
    source = Field(tokenize='spacy', tokenizer_language = 'de_core_news_sm', init_token='<sos>', eos_token='<eos>', lower=True)
    target = Field(tokenize='spacy', tokenizer_language = 'en_core_web_sm', init_token='<sos>', eos_token='<eos>', lower=True)

    # Splits the data in Train, Test and Validation data
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(source, target))

    # Build the vocabulary for both the language
    source.build_vocab(train_data, min_freq=3)
    target.build_vocab(train_data, min_freq=3)

    # Create the Iterator using builtin Bucketing
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                          batch_size=batch_size,
                                                                          sort_within_batch=True,
                                                                          sort_key=lambda x: len(x.src),
                                                                          device=device)
    return train_iterator, valid_iterator, test_iterator, source, target

class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input

class Integrate_Neuron(nn.Module):
    def __init__(self, input_dim):
        super(Integrate_Neuron, self).__init__()
        self.input_dim  	= input_dim
        self.mem        	= torch.zeros(input_dim).to(device)
        self.reset_mem()

    def reset_mem(self):
        self.mem = torch.zeros(self.input_dim).to(device)
        
    def forward(self, delta_mem):
        #pdb.set_trace()
        self.mem = self.mem + delta_mem
        return self.mem

class LIF(nn.Module):
	def __init__(self, input_dim, threshold=1.0, leak=0.95):
		super(LIF, self).__init__()
		self.input_dim 		= input_dim
		self.mem 			= torch.Tensor(input_dim).to(device)
		self.threshold 		= nn.Parameter(torch.tensor(threshold))
		self.leak 			= nn.Parameter(torch.tensor(leak))
		self.act_func 		= LinearSpike.apply
		#self.batch_norm		= nn.BatchNorm1d(input_dim)
		self.max_mem 		= torch.tensor(0.0)
		self.min_mem 		= torch.tensor(0.0)
		self.reset_mem()
		
	def reset_mem(self):
		self.mem = torch.zeros(self.input_dim).to(device)
		self.leak.data = self.leak.clamp(min=0.0, max=1.0)
		
	def set_threshold(self, threshold):
		self.threshold.data = torch.tensor(threshold)

	def set_leak(self, leak):
		self.leak.data       = torch.tensor(leak)
	
		
	def forward(self, delta_mem):
		#pdb.set_trace()
		self.mem = self.leak*self.mem + delta_mem
		#self.mem = self.leak*self.mem + self.batch_norm(delta_mem)
		#self.mem = self.batch_norm(self.mem)
		mem_thr = self.mem/self.threshold - 1.0
		rst = self.threshold*(mem_thr>0).float()
		self.mem = self.mem - rst
		out = self.act_func(mem_thr)
		self.max_mem = self.mem.max()
		self.min_mem = self.mem.min()
		return out

class SpikeEncoder(nn.Module):
    def __init__(self, vocab_len, embedding_dim, hidden_dim, dropout_prob, timesteps=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_len, embedding_dim)
        self.snn = nn.Sequential(
							nn.Linear(embedding_dim, hidden_dim),
							LIF(input_dim=hidden_dim),
							nn.Dropout(dropout_prob),
							nn.Linear(hidden_dim, hidden_dim),
							Integrate_Neuron(input_dim=hidden_dim)
							)
        self.timesteps = timesteps
        self.dropout = nn.Dropout(dropout_prob)
    
    def initialize_mask(self):
        self.mask = {}
        prev_layer = None
        for l in self.snn.named_children():
            if isinstance(l[1], nn.Dropout):
                self.mask[int(l[0])] = l[1](torch.ones(prev_layer.mem.shape)).to(device)
            prev_layer = l[1]
    
    def reset_LIF(self):
        for l in self.snn.named_children():
        	if isinstance(l[1], (LIF, Integrate_Neuron)):
        		l[1].reset_mem()
    
    def forward(self, input_batch):
        
        self.reset_LIF()
        self.initialize_mask()
        embed = self.dropout(self.embedding(input_batch))
        for word in range(embed.shape[0]):
            for t in range(self.timesteps):
                out = embed[word,:,:]
                for l in self.snn.named_children():
                    if isinstance(l[1], nn.Dropout):
                    	out = out*self.mask[int(l[0])]
                    elif isinstance(l[1], LIF):
                    	out = l[1](out)
                    else:
                    	out = l[1](out)
        out = out/embed.shape[1]
        return out

class Encoder(nn.Module):
    def __init__(self, vocab_len, embedding_dim, hidden_dim, n_layers, dropout_prob):
        super().__init__()

        self.embedding = nn.Embedding(vocab_len, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_prob)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_batch):
        embed = self.dropout(self.embedding(input_batch))
        outputs, (hidden, cell) = self.rnn(embed)

        return hidden, cell

class SpikeOneStepDecoder(nn.Module):
    def __init__(self, input_output_dim, embedding_dim, hidden_dim, dropout_prob, timesteps=10):
        super().__init__()
        self.input_output_dim = input_output_dim
        self.embedding = nn.Embedding(input_output_dim, embedding_dim)
        self.snn = nn.Sequential(
							nn.Linear(embedding_dim, hidden_dim),
							LIF(input_dim=hidden_dim),
							nn.Dropout(dropout_prob),
							nn.Linear(hidden_dim, hidden_dim),
							Integrate_Neuron(input_dim=hidden_dim)
							)
        self.fc = nn.Linear(hidden_dim, input_output_dim)
        self.timesteps = timesteps
        self.dropout = nn.Dropout(dropout_prob)

    def initialize_mask(self):
        self.mask = {}
        prev_layer = None
        for l in self.snn.named_children():
            if isinstance(l[1], nn.Dropout):
                self.mask[int(l[0])] = l[1](torch.ones(prev_layer.mem.shape)).to(device)
            prev_layer = l[1]
    
    def reset_LIF(self):
        for l in self.snn.named_children():
        	if isinstance(l[1], (LIF, Integrate_Neuron)):
        		l[1].reset_mem()
    
    def forward(self, target_token, hidden):
        target_token = target_token.unsqueeze(0)
        self.reset_LIF()
        self.initialize_mask()
        self.snn[1].mem = hidden
        target_token = target_token.unsqueeze(0)
        embedding_layer = self.dropout(self.embedding(target_token))
        for word in range(embedding_layer.shape[0]):
            for t in range(self.timesteps):
                out = embedding_layer[word,:,:]
                for l in self.snn.named_children():
                    if isinstance(l[1], nn.Dropout):
                    	out = out*self.mask[int(l[0])]
                    elif isinstance(l[1], LIF):
                    	out = l[1](out)
                    else:
                    	out = l[1](out)
        out = out/embedding_layer.shape[1]
        linear = self.fc(out.squeeze(0))
        return linear, out

class OneStepDecoder(nn.Module):
    def __init__(self, input_output_dim, embedding_dim, hidden_dim, n_layers, dropout_prob):
        super().__init__()
        # self.input_output_dim will be used later
        self.input_output_dim = input_output_dim

        self.embedding = nn.Embedding(input_output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, input_output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, target_token, hidden, cell):
        target_token = target_token.unsqueeze(0)
        embedding_layer = self.dropout(self.embedding(target_token))
        output, (hidden, cell) = self.rnn(embedding_layer, (hidden, cell))

        linear = self.fc(output.squeeze(0))

        return linear, hidden, cell

class SpikeDecoder(nn.Module):
    def __init__(self, one_step_decoder, device):
        super().__init__()
        self.one_step_decoder = one_step_decoder
        self.device = device

    def forward(self, target, hidden, teacher_forcing_ratio=0.5):
        target_len, batch_size = target.shape[0], target.shape[1]
        target_vocab_size = self.one_step_decoder.input_output_dim
        # Store the predictions in an array for loss calculations
        predictions = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        # Take the very first word token, which will be sos
        input = target[0, :]

        # Loop through all the time steps, starts from 1
        for t in range(1, target_len):
            predict, hidden = self.one_step_decoder(input, hidden)

            predictions[t] = predict
            input = predict.argmax(1)

            # Teacher forcing
            do_teacher_forcing = random.random() < teacher_forcing_ratio
            input = target[t] if do_teacher_forcing else input

        return predictions

class Decoder(nn.Module):
    def __init__(self, one_step_decoder, device):
        super().__init__()
        self.one_step_decoder = one_step_decoder
        self.device = device

    def forward(self, target, hidden, cell, teacher_forcing_ratio=0.5):
        target_len, batch_size = target.shape[0], target.shape[1]
        target_vocab_size = self.one_step_decoder.input_output_dim
        # Store the predictions in an array for loss calculations
        predictions = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        # Take the very first word token, which will be sos
        input = target[0, :]

        # Loop through all the time steps, starts from 1
        for t in range(1, target_len):
            predict, hidden, cell = self.one_step_decoder(input, hidden, cell)

            predictions[t] = predict
            input = predict.argmax(1)

            # Teacher forcing
            do_teacher_forcing = random.random() < teacher_forcing_ratio
            input = target[t] if do_teacher_forcing else input

        return predictions

class SpikeEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_forcing_ratio= 0.5):
        hidden = self.encoder(source)
        outputs = self.decoder(target, hidden, teacher_forcing_ratio)
        return outputs

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        hidden, cell = self.encoder(source)
        outputs = self.decoder(target, hidden, cell, teacher_forcing_ratio)

        return outputs


def create_model(source, target):
    # Define the required dimensions and hyper parameters
    embedding_dim = 256
    hidden_dim = 1024
    dropout = 0.2
    
    # Instanciate the models
    encoder = Encoder(len(source.vocab), embedding_dim, hidden_dim, n_layers=2, dropout_prob=dropout)
    
    one_step_decoder = OneStepDecoder(len(target.vocab), embedding_dim, hidden_dim, n_layers=2, dropout_prob=dropout)
    
    decoder = Decoder(one_step_decoder, device)

    model = EncoderDecoder(encoder, decoder)
    
    model = model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters())

    # Makes sure the CrossEntropyLoss ignores the padding tokens.
    TARGET_PAD_IDX = target.vocab.stoi[target.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_IDX)

    return model, optimizer, criterion

def create_spiking_model(source, target):
    # Define the required dimensions and hyper parameters
    embedding_dim = 256
    hidden_dim = 1024
    dropout = 0.2
    
    encoder = SpikeEncoder(len(source.vocab), embedding_dim, hidden_dim, dropout_prob=dropout)
    
    one_step_decoder = SpikeOneStepDecoder(len(target.vocab), embedding_dim, hidden_dim, dropout_prob=dropout)
    
    decoder = SpikeDecoder(one_step_decoder, device)

    model = SpikeEncoderDecoder(encoder, decoder)

    model = model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters())

    # Makes sure the CrossEntropyLoss ignores the padding tokens.
    TARGET_PAD_IDX = target.vocab.stoi[target.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_IDX)

    return model, optimizer, criterion


def train(train_iterator, valid_iterator, source, target, epochs=10):
    model, optimizer, criterion = create_model(source, target)

    #model, optimizer, criterion = create_spiking_model(source, target)

    clip = 1

    for epoch in range(1, epochs + 1):
        pbar = tqdm(total=len(train_iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', unit=' batches', ncols=200)

        training_loss = []
        # set training mode
        model.train()

        # Loop through the training batch
        for i, batch in enumerate(train_iterator):
            # Get the source and target tokens
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()

            # Forward pass
            
            output = model(src, trg)

            # reshape the output
            output_dim = output.shape[-1]

            # Discard the first token as this will always be 0
            output = output[1:].view(-1, output_dim)

            # Discard the sos token from target
            trg = trg[1:].view(-1)

            # Calculate the loss
            loss = criterion(output, trg)

            # back propagation
            loss.backward()

            # Gradient Clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            training_loss.append(loss.item())

            pbar.set_postfix(
                epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}", refresh=True)
            pbar.update()

        with torch.no_grad():
            # Set the model to eval
            model.eval()

            validation_loss = []

            # Loop through the validation batch
            for i, batch in enumerate(valid_iterator):
                src = batch.src
                trg = batch.trg

                # Forward pass
                output = model(src, trg, 0)

                output_dim = output.shape[-1]

                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                # Calculate Loss
                loss = criterion(output, trg)

                validation_loss.append(loss.item())

        pbar.set_postfix(
            epoch=f" {epoch}, train loss= {round(sum(training_loss) / len(training_loss), 4)}, val loss= {round(sum(validation_loss) / len(validation_loss), 4)}",
            refresh=False)
        pbar.close()

    return model


if __name__ == '__main__':
    train_iterator, valid_iterator, test_iterator, source, target = get_datasets(batch_size=512)
    
    model = train(train_iterator, valid_iterator, source, target, epochs=25)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'source': source.vocab,
        'target': target.vocab
    }

    torch.save(checkpoint, 'nmt-model-lstm-25.pth')
