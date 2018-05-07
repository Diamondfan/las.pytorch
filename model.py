#encoding=utf-8

import math
import torch
import random
import editdistance as ed
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from attention import GlobalAttention
from decoder import edit_distance

RNNs = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
Activates = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}

class SequenceWise(nn.Module):
    "3-D matrix can use module that only support 2-D matrix such as nn.Linear"
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t*n, -1)
        x = self.module(x)            
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class BatchRNN(nn.Module):
    """
    Add BatchNorm before rnn to generate a batchrnn layer
    """
    def __init__(self, input_size, rnn_param, batch_norm=True, dropout=0.1):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = rnn_param["hidden_size"]
        self.bidirectional = rnn_param["bidirectional"]
        rnn_type = RNNs[rnn_param["rnn_type"]]
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=self.hidden_size, bidirectional=self.bidirectional, dropout=dropout, bias=False)
        
    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        #self.rnn.flatten_parameters()
        return x

class LayerCNN(nn.Module):
    """
    One CNN layer include conv2d, batchnorm, activation and maxpooling
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, pooling_size=None, 
                        activation_function=nn.ReLU, batch_norm=True):
        super(LayerCNN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channel) if batch_norm else None
        self.activation = activation_function(inplace=True)
        if pooling_size is not None:
            self.pooling = nn.MaxPool2d(pooling_size)
        else:
            self.pooling = None
        
    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.pooling is not None:
            x = self.pooling(x)
        return x

class Encoder(nn.Module):
    def __init__(self, rnn_param=None, cnn_param=None, drop_out=0.1):
        """
        rnn_param:    the dict of rnn parameters               type dict
            rnn_param = {"rnn_input_size":40, "rnn_hidden_size":256, ....}
        cnn_param:    the cnn parameters, only support Conv2d  type list
            cnn_param = {"layer":[[(in_channel, out_channel), (kernel_size), (stride), (padding), (pooling_size)],...], 
                            "batch_norm":True, "activate_function":nn.ReLU}
        drop_out :    drop_out paramteter for all place where need drop_out
        """
        super(Encoder, self).__init__()
        if rnn_param is None or type(rnn_param) != dict:
            raise ValueError("rnn_param need to be a dict to contain all params of rnn!")
        self.rnn_param = rnn_param
        self.cnn_param = cnn_param
        self.num_directions = 2 if rnn_param["bidirectional"] else 1
        self._drop_out = drop_out
        
        rnn_input_size = rnn_param["rnn_input_size"]
        cnns = []
        activation = Activates[cnn_param["activate_function"]]
        batch_norm = cnn_param["batch_norm"]
    
        cnn_layers = len(cnn_param["layers"])
        for layer in range(cnn_layers):
            in_channel = eval(cnn_param["layers"][layer][0])[0]
            out_channel = eval(cnn_param["layers"][layer][0])[1]
            kernel_size = eval(cnn_param["layers"][layer][1])
            stride = eval(cnn_param["layers"][layer][2])
            padding = eval(cnn_param["layers"][layer][3])
            pooling_size = eval(cnn_param["layers"][layer][4])
        
            cnn = LayerCNN(in_channel, out_channel, kernel_size, stride, padding, pooling_size, 
                        activation_function=activation, batch_norm=batch_norm)
            cnns.append(('%d' % layer, cnn))
       
            rnn_input_size = int(math.floor((rnn_input_size+2*padding[1]-kernel_size[1])/stride[1])+1)    
        self.conv = nn.Sequential(OrderedDict(cnns))
        rnn_input_size *= out_channel
        
        rnns = [] 
        rnn = BatchRNN(rnn_input_size, rnn_param, batch_norm=False, dropout=drop_out)
        rnns.append(('0', rnn))
        rnn_layers = rnn_param["layers"]
        for i in range(1, rnn_layers):
            rnn = BatchRNN(self.num_directions*rnn.hidden_size, rnn_param, batch_norm=rnn_param["batch_norm"], dropout=drop_out)
            rnns.append(('%d' % i, rnn))

        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.hidden_size = rnn.hidden_size * self.num_directions

    def forward(self, x):
        #x: batch_size * max_seq_length * feat_size
        x = x.unsqueeze(1)
        x = self.conv(x)
                
        x = x.transpose(1, 2).contiguous()        
        sizes = x.size()
        x = x.view(sizes[0], sizes[1], sizes[2]*sizes[3])
        x = x.transpose(0, 1).contiguous()
        
        x = self.rnns(x)
        return x

class LAS(nn.Module):
    def __init__(self, vocab, encoder_param, decoder_param):
        super(LAS, self).__init__()
        self.vocab = vocab
        vocab_size = vocab.n_words
        self.encoder_param = encoder_param
        self.decoder_param = decoder_param
        
        rnn_param = encoder_param["rnn"]
        cnn_param = encoder_param["cnn"]
        drop_out = encoder_param["drop_out"]
        self.encoder = Encoder(rnn_param, cnn_param, drop_out=drop_out)

        embed_dim = decoder_param["embed_dim"]
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        hidden_dim = self.encoder.hidden_size
        self.rnn = nn.LSTM(input_size=embed_dim+hidden_dim, hidden_size=hidden_dim, num_layers=decoder_param["rnn_layers"], dropout=decoder_param["drop_out"])

        self.attention = GlobalAttention(mode=self.decoder_param["attention_mode"])
        self.teach_rate = decoder_param.get("teach_rate", 0)
        self.teach_forcing = (self.teach_rate != 0)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        linear = nn.Sequential(nn.Linear(hidden_dim * 2, 512),
                                nn.ReLU(),
                                nn.Linear(512, vocab_size))
        self.fc = SequenceWise(linear)

    def forward(self, x, y=None, max_step=40):
        """
        x  : batch_size * seq_len * feat_size
        y  : batch_size * seq_len
        """
        x = self.encoder(x)
        x = x.transpose(0, 1)
        batch_size = x.size(0)
        
        if type(y) != type(None):
            inputs = self.embedding(y)
        
        sos = Variable(torch.zeros(batch_size, 1).long())
        if x.is_cuda:
            iy = self.embedding(sos.cuda())
        else:
            iy = self.embedding(sos)
        iy = torch.cat((iy, x[:,0:1,:]), dim=-1)

        step = y.size(1) if type(y) != type(None) else max_step
        out = []
        aligns = []
        hx = None
        for t in range(step):
            sx, hx = self.rnn(iy, hx=hx)
            ctx, ax = self.attention(x, sx)
            aligns.append(ax)
            out.append(self.fc(torch.cat((sx, ctx), dim=-1)))
            if self.teach_forcing and random.random() < self.teach_rate:
                iy = inputs[:, t:t+1, :]
            else:
                iy = torch.max(out[-1], dim=2)[1]
                iy = self.embedding(iy)

            iy = torch.cat((iy, ctx), dim=-1)

        out = torch.cat(out, dim=1)
        aligns = torch.stack(aligns, dim=1)
        return out, aligns 

    def set_eval(self):
        "set mode for validation and testing"
        self.eval()
        self.volatile = True
        self.teach_forcing = False

    def set_train(self):
        "set mode for training"
        self.train()
        self.volatile = False
        self.teach_forcing = (self.teach_rate != 0)

    def loss(self, out, y):
        batch_size, _, out_dim = out.size()
        out = out.view(-1, out_dim)
        y = y.view(-1)
        loss = self.loss_fn(out, y)
        return loss
    
    def cer_batch(self, out, y, eos=1):
        probs = F.softmax(out, dim=2)
        seq = torch.max(probs, 2)[1].data.cpu().numpy()
        y = y.cpu().data.numpy()
        num_err = 0
        num_words = 0
        for x in range(len(seq)):
            batch_y = [w for w in y[x] if (w != 0 and w != eos)]
            batch_s = []
            for char in seq[x]:
                if char == 0:
                    continue
                if char == eos:
                    break
                batch_s.append(char)
            num_err += ed.eval(batch_s, batch_y)
            num_words += len(batch_y)
        return num_err, num_words

    @staticmethod
    def save_package(model, optimizer=None, train_loss_results=None, train_cer_results=None, cv_loss_results=None, cv_cer_results=None):
        package = {
                'encoder_param': model.encoder_param,
                'decoder_param': model.decoder_param,
                'vocab': model.vocab,
                'state_dict': model.state_dict()
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        package["train_loss_results"] = train_loss_results
        package["train_cer_results"] = train_cer_results
        package["cv_loss_results"] = cv_loss_results
        package["cv_cer_results"] = cv_cer_results
        return package

