import argparse
import torch
import torchtext
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
import os
import dill
import pdb
import time
import spacy
from matplotlib import pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class PoissonGenerator(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self,input):
        
        out = torch.mul(torch.le(torch.rand_like(input), torch.abs(input)*1.0).float(),torch.sign(input))
        return out

class LSTM(nn.Module):
	def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx, num_directions=1):
		super().__init__()
		self.embedding 	= nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
		self.lstm 		= nn.LSTM(embedding_dim, hidden_dim, bidirectional=True if num_directions>1 else False, num_layers=n_layers)
		self.fc 		= nn.Linear(hidden_dim*num_directions, output_dim)
		self.dropout 	= nn.Dropout(dropout)

	def forward(self, text, text_lengths):
		#pdb.set_trace()
		embedded = self.dropout(self.embedding(text))
		embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
		packed_output, (hidden, cell) = self.lstm(embedded)
		#pdb.set_trace()
		hidden = hidden.view(N_LAYERS, N_DIRECTIONS, text.shape[1], HIDDEN_DIM)
		hidden = hidden[-1,:,:,:]
		if N_DIRECTIONS>1:
			hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
		else:
			hidden = hidden.squeeze(0)
		hidden = self.dropout(hidden)
		#output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
		out = self.fc(hidden)
		return out

class RNN(nn.Module):
	def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx, num_directions=1):
		super().__init__()
		self.embedding 	= nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
		self.rnn 		= nn.RNN(embedding_dim, hidden_dim, bidirectional=True if num_directions>1 else False, num_layers=n_layers)
		self.fc 		= nn.Linear(hidden_dim*num_directions, output_dim)
		self.dropout 	= nn.Dropout(dropout)

	def forward(self, text, text_lengths):
		#pdb.set_trace()
		embedded = self.dropout(self.embedding(text))
		embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
		output, hidden = self.rnn(embedded)
		hidden = hidden.view(N_LAYERS, N_DIRECTIONS, text.shape[1], HIDDEN_DIM)
		hidden = hidden[-1,:,:,:]
		if N_DIRECTIONS>1:
			hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
		else:
			hidden = hidden.squeeze(0)
		hidden = self.dropout(hidden)
		#pdb.set_trace()
		#output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
		#hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
		out = self.fc(hidden)
		return out

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

class SNN(nn.Module):
	def __init__(self, input_dim=100, embedding_dim=100, hidden_dim=128, output_dim=1, n_layers=1, n_directions=1, dropout=0.2, pad_idx=0, timesteps=10):
		super().__init__()
		self.embedding 	= nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
		self.snn 		= nn.Sequential(
							nn.Linear(embedding_dim, hidden_dim),
							LIF(input_dim=hidden_dim),
							nn.Dropout(0.2),
							nn.Linear(hidden_dim, hidden_dim),
							LIF(input_dim=hidden_dim),
							nn.Dropout(0.2),
							nn.Linear(hidden_dim, hidden_dim),
							Integrate_Neuron(input_dim=hidden_dim)
							)
		self.ann 		= nn.Sequential(
							nn.Linear(hidden_dim, hidden_dim),
							nn.ReLU(inplace=True),
							nn.Dropout(0.5),
							nn.Linear(hidden_dim, output_dim)
							)

		self.timesteps 	= timesteps
		self.dropout 	= nn.Dropout(dropout)
		self.n_directions = n_directions
		if self.n_directions>1:
			self.bidirec_linear = nn.Linear(2, output_dim)
		self._initialize_weights()
		
	def initialize_mask(self):
		self.mask = {}
		prev_layer = None
		for l in self.snn.named_children():
			if isinstance(l[1], nn.Dropout):
				self.mask[int(l[0])] = l[1](torch.ones(prev_layer.mem.shape)).to(device)
			prev_layer = l[1]

	def _initialize_weights(self):
		for l in self.snn.named_children():
			m = l[1]
			
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

			#elif isinstance(m, LIF):
			#	m.batch_norm.weight.data.fill_(1)
			#	m.batch_norm.bias.data.zero_()

			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()
		for l in self.ann.named_children():
			if isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()
		
		if self.n_directions>1:
			self.bidirec_linear.weight.data.normal_(0, 0.01)
			self.bidirec_linear.bias.data.zero_()

	def reset_LIF(self):
		for l in self.snn.named_children():
			if isinstance(l[1], (LIF, Integrate_Neuron)):
				l[1].reset_mem()
	
	def forward(self, text, text_lengths, track_vmem=False):
		for direction in range(self.n_directions):
			self.reset_LIF()
			self.initialize_mask()
			if track_vmem:
				self.tracked_vmem = []
			embedded = self.dropout(self.embedding(text))
			for word in range(embedded.shape[0]):
				for t in range(self.timesteps):
					if direction==0:
						out = embedded[word,:,:]
					elif direction==1:
						out = embedded[-(word+1),:,:]
					for l in self.snn.named_children():
						if isinstance(l[1], nn.Dropout):
							out = out*self.mask[int(l[0])]
						elif isinstance(l[1], LIF):
							out = l[1](out)
						else:
							out = l[1](out)
					if track_vmem:
						self.tracked_vmem.append(out.item())
				if direction==0:
					out1 = out.clone()
				elif direction==1:
					out2 = out.clone()
		if self.n_directions==1:
			out = out1
		elif self.n_directions==2:
			#print((out1==out2).sum())
			out = self.bidirec_linear(torch.stack((out1, out2), dim=1).squeeze(-1))
		
		out = self.ann(out/embedded.shape[1])
		return out

def time_difference(start_time, end_time):
	out = ''
	diff_sec = int(end_time - start_time)
	if diff_sec>=60:
		minutes = int(diff_sec//60)
		seconds = int(diff_sec%60)
		out = str(minutes)+'m '+str(seconds)+'s'
	else:
		out = str(diff_sec)+'s'
		minutes=0
	if minutes>=60:
		hours = int(minutes//60)
		minutes = int(minutes%60)
		out = str(hours)+'h '+str(minutes)+'m '+str(seconds)+'s'
	return out

def binary_accuracy(pred, label):
	pred = torch.round(torch.sigmoid(pred))
	correct = (pred==label).float()
	acc = correct.sum()/len(correct)
	return correct.sum().item()

def save_model():
	state = {
			'accuracy' 	: best_acc,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict()
			}

	try:
		os.mkdir('./trained_models/')
	except OSError:
		pass
	
	filename = './trained_models/'+identifier+'.pth'
	if not args.dont_save:
		torch.save(state, filename)

def plot_tracked_vmem(vmem, sentence):
	plt.plot(vmem, label=sentence)
	plt.legend()
	plt.xlabel('Timesteps')
	plt.ylabel('Vmem')
	plt.grid()
	plt.pause(0.1)

def run_custom_test(inputs):
	for s in inputs:
		print(f'\n{s}:{custom_test(s)}')
		print(model.tracked_vmem)
		plot_tracked_vmem(model.tracked_vmem, s)
	plt.show()

def custom_test(sentence=''):
	model.eval()
	nlp = spacy.load('en_core_web_sm')
	tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
	indexed = [TEXT.vocab.stoi[t] for t in tokenized]
	length = [len(indexed)]
	tensor = torch.LongTensor(indexed).to(device)
	tensor = tensor.unsqueeze(0)
	length_tensor = torch.LongTensor(length)
	prediction = torch.sigmoid(model(tensor, track_vmem=True))
    
	return prediction.item()

def train():
	model.train()
	losses 	= AverageMeter('Loss')
	acc  	= AverageMeter('Accuracy')
	
	for param_group in optimizer.param_groups:
		lr = param_group['lr']	
	
	for batch_idx, data in enumerate(train_iterator):
		
		optimizer.zero_grad()
		text, text_lengths = data.text
		output = model(text, text_lengths).squeeze(1)
		loss = criterion(output, data.label)
		#make_dot(loss).view()
		#exit()
		loss.backward()
		optimizer.step()
		
		pred = torch.round(torch.sigmoid(output))
		correct = (pred==data.label).cpu().sum()
		losses.update(loss.item(), text.shape[1])
		acc.update(correct.item()/text.shape[1], text.shape[1])
		
		thresholds = []
		leaks = []
		vmem_min = []
		vmem_max = []
		if mode=='snn':
			for l in model.module.snn.named_children():
				if isinstance(l[1], LIF):
					thresholds.append(round(l[1].threshold.item(), 2))
					leaks.append(round(l[1].leak.item(), 2))
					vmem_min.append(round(l[1].min_mem.item(), 2))
					vmem_max.append(round(l[1].max_mem.item(), 2))
			if batch_idx%50==0:
				print(f'\nBatch:{batch_idx+1}, Loss:{losses.avg:.4f}, Acc:{acc.avg:.4f}, Thresholds:{thresholds}, Leaks:{leaks}, min_mem:{vmem_min}, max_mem:{vmem_max} Time:{time_difference(start_time, time.time())}')
		#if batch_idx==250:
		#	break

	print(f'Epoch:{epoch}, lr:{lr:.2e}, train_loss:{losses.avg:.4f}, train_acc:{acc.avg:.4f}, train_time:{time_difference(start_time, time.time())}', end=', ')

def test():
	model.eval()
	losses = AverageMeter('Loss')
	acc = AverageMeter('Accuracy')
	global best_acc
	with torch.no_grad():
		for batch_idx, data in enumerate(test_iterator):
			text, text_lengths = data.text
			output = model(text, text_lengths).squeeze(1)
			loss = criterion(output, data.label)
			pred = torch.round(torch.sigmoid(output))
			correct = (pred==data.label).cpu().sum()
			losses.update(loss.item(), text.shape[1])
			acc.update(correct.item()/text.shape[1], text.shape[1])
			if mode=='snn' and batch_idx%10==0 and args.test_only:
				print(f'Batch:{batch_idx}, Acc:{acc.avg:.4f}')
			#if batch_idx==10:
			#	break

		
		if acc.avg>best_acc:
			best_acc = acc.avg
			state = {
					'accuracy' 	: best_acc,
					'epoch' 	: epoch,
					'state_dict': model.module.state_dict(),
					'optimizer' : optimizer.state_dict()
			}

			try:
				os.mkdir('./trained_models/')
			except OSError:
				pass
	
			filename = './trained_models/'+identifier+'.pth'
			if not args.dont_save:
				torch.save(state, filename)

		print(f'test_loss:{losses.avg:.4f}, test_acc:{acc.avg:.4f}, best_acc:{best_acc:.4f}, epoch_time:{time_difference(start_time, time.time())}')

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Train ANN to be later converted to SNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--mode',         	default='rnn',	type=str,	help='rnn, snn, lstm',	choices=['rnn', 'snn', 'lstm'])
	parser.add_argument('--hidden_dim',   	default=256,	type=int,	help='number of neurons in hidden dimension')
	parser.add_argument('--n_layers',	  	default=1,		type=int,	help='number of hidden layers')
	parser.add_argument('--n_directions', 	default=1,		type=int,   help='bidirectional=2 else 1',	choices=[1,2])
	parser.add_argument('--dropout', 		default=0.5,	type=float,	help='dropout')
	parser.add_argument('--learning_rate', 	default=1e-3,	type=float,	help='learning rate')
	parser.add_argument('--timesteps',      default=5,		type=int,	help='timesteps for snn')
	parser.add_argument('--epochs',      	default=50,		type=int,	help='number of training epochs')
	parser.add_argument('--devices',        default='3',    type=str,   help='list of gpu device(s)')
	parser.add_argument('--dont_save',		action='store_true',		help='don\'t save training model during testing')
	parser.add_argument('--test_only',      action='store_true',        help='perform only inference')
	parser.add_argument('--pretrained_rnn', default='',     type=str,	help='pretrained model to initialize ANN')
	parser.add_argument('--pretrained_snn', default='',     type=str,	help='pretrained SNN model for evaluation')
	parser.add_argument('--custom_examples',     	default='',               nargs='+',     	help='list of strings for evaluation')
	
	args=parser.parse_args()
	
	os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	SEED 	= 1234
	torch.manual_seed(SEED)

	TEXT 	= data.Field(tokenize='spacy', tokenizer_language = 'en_core_web_sm', include_lengths=True, batch_first=False)
	LABEL 	= data.LabelField(dtype=torch.float)
	start_time = time.time()
	train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root='/local/a/rathi2/Datasets/')
	print(f'Time to load dataset - {time_difference(start_time, time.time())}')
	
	MAX_VOCAB_SIZE 	= 25_000
	TEXT.build_vocab(train_data,
					 max_size=MAX_VOCAB_SIZE,
					 vectors = "glove.6B.100d",
					 vectors_cache ='/local/a/rathi2/vector_cache',
					 unk_init = torch.Tensor.normal_
					 )
	LABEL.build_vocab(train_data)
	
	BATCH_SIZE 		= 64
	INPUT_DIM 		= len(TEXT.vocab)
	EMBEDDING_DIM 	= 100
	HIDDEN_DIM 		= args.hidden_dim
	OUTPUT_DIM 		= 1
	N_LAYERS 		= args.n_layers
	N_DIRECTIONS 	= args.n_directions
	DROPOUT 		= args.dropout
	N_EPOCHS 		= args.epochs
	LEARNING_RATE 	= args.learning_rate
	best_acc 		= 0.0
	mode 			= args.mode #['snn', 'rnn', 'lstm']
	pretrained_snn 	= args.pretrained_snn
	pretrained_rnn	= args.pretrained_rnn
	timesteps 		= args.timesteps
		
	identifier 	= mode+str(N_DIRECTIONS)+'_sentiment_analysis'

	print('Run on time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            
	print('\n Arguments:')
	for arg in vars(args):
		print('\t {:20} : {}'.format(arg, getattr(args,arg)))

	train_iterator, test_iterator = data.BucketIterator.splits((train_data, test_data), batch_size=BATCH_SIZE, device=device, sort_within_batch = True)
	
	PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
	
	if mode=='rnn':
		model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, PAD_IDX, N_DIRECTIONS)
	if mode=='lstm':
		model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, PAD_IDX, N_DIRECTIONS)
	elif mode == 'snn':
		model = SNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, N_DIRECTIONS, DROPOUT, PAD_IDX, timesteps=timesteps)
	
	model = model.to(device)
	print(model)
	#pdb.set_trace()
	pretrained_embeddings = TEXT.vocab.vectors
	model.embedding.weight.data.copy_(pretrained_embeddings)
	
	UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
	
	model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
	model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
	
	if pretrained_snn or pretrained_rnn:
		if pretrained_snn:
			state = torch.load(pretrained_snn, map_location='cpu')
		else:
			state = torch.load(pretrained_rnn, map_location='cpu')
		missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
		best_acc = state['accuracy']
		print(f'Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}')
		print(f'Info: Accuracy of loaded model: {best_acc:.4f}')

	if len(args.custom_examples)>0:
		run_custom_test(args.custom_examples)
		exit()
	
	#model = nn.DataParallel(model)

	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
	print(optimizer)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=N_EPOCHS//3, gamma=0.2, verbose=False)
	criterion = nn.BCEWithLogitsLoss()
	criterion.to(device)
	
	parameter_count = 0
	for param in model.snn.parameters():
	    parameter_count+=param.numel()
	print(f'Trainable parameters: {parameter_count}')
	exit()

	best_acc = 0.0
	for epoch in range(N_EPOCHS):
		start_time = time.time()
		if args.test_only:
			test()
			break
		train()
		test()
		scheduler.step()
	print(f'Best test accuracy:{best_acc:.4f}, args:{args}')
