#encoding=utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable

import copy
import time
import yaml
import argparse
import numpy as np

from model import *
from data import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser("Training LAS model")
parser.add_argument("--config", default="./conf/LAS_config.yaml", help="Config file of training LAS MODEL.")


def run_epoch(model, loader, optimizer=None, is_training=True, print_every=200, USE_CUDA=True):
    if is_training:
        model.set_train()
    else:
        model.set_eval()
    
    total_loss = 0
    num_errs = 0
    num_words = 0
    i = 1
    for data in loader:
        inputs, targets = data
 
        inputs = Variable(inputs, volatile=model.volatile)
        targets = Variable(targets, volatile=model.volatile, requires_grad=False)

        if USE_CUDA:
            inputs = inputs.cuda()
            targets = targets.cuda()

        if is_training:
            out, _ = model(inputs, y=targets)
        else:
            max_lab_len = targets.size(1)
            out, _ = model(inputs, y=None, max_step=max_lab_len)
        loss = model.loss(out, targets)
        
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), 400)
            optimizer.step()
        total_loss += loss.data[0]

        errs, words = model.cer_batch(out, targets, eos=EOS)
        num_errs += errs
        num_words += words

        if i % print_every == 0 and is_training:
            cer = (num_errs / float(num_words)) * 100
            print('batch = %d, loss = %.4f, cer = %.2f' % (i, total_loss / i, cer), flush=True)
        i += 1
    average_loss = total_loss / (i - 1)
    cer = (num_errs / float(num_words)) * 100
    return average_loss, cer
    
def main():
    args = parser.parse_args()
    config_file = args.config
    cf = yaml.load(open(config_file, 'r'))
    
    USE_CUDA = cf["train"]["use_cuda"]
    try:
        seed = cf["train"]["seed"]
    except:
        seed = torch.initial_seed()
    
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)

    log_file = open(os.path.join(cf["data"]["log"], cf['data']['exp']+'.log'), 'w')
    
    ###print arguments setting
    for key in cf:
        print('{}:'.format(key), file=log_file)
        for para in cf[key]:
            print('{:50}:{}'.format(para,cf[key][para]))
            print('{:50}:{}'.format(para,cf[key][para]),file=log_file)
        print('\n',file=log_file)
    print('please check.',flush=True,file=log_file) 
    
    print("------Generate Vocabulary and Dataset------")
    checkpoints = cf['data']['checkpoints']
    data_dir = cf["data"]["data_dir"]
    vocab_file = cf["data"]["vocab_file"]
    batch_size = cf["data"]["batch_size"]
    num_workers = cf["data"]["num_workers"]
    max_lab_len = cf['data']['max_lab_len']

    vocab = read_label(vocab_file)
    train_dataset = SpeechDataset(data_dir, dataset="train", vocab=vocab, lab_len=max_lab_len)
    cv_dataset = SpeechDataset(data_dir, dataset="cv", vocab=vocab, lab_len=max_lab_len)
    train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    cv_loader = SpeechDataLoader(cv_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("numbers of words in vocab is: %d" % vocab.n_words)

    #Define the model
    encoder_param = cf["model"]["encoder_param"]
    decoder_param = cf["model"]["decoder_param"]
    model = LAS(vocab, encoder_param, decoder_param)
    for idx, m in enumerate(model.children()):
        print(idx, m)
        print(idx, m, file=log_file)
                            
    if USE_CUDA:
        model.cuda()
        #model = nn.DataParallel(model, device_ids=device_ids)

    print("----------Start Training----------")
    
    epochs = cf["train"]["epoch"]
    decay_epoch = cf['train']['decay_epoch']
    init_lr = cf["train"]["init_lr"]
    weight_decay = cf["train"]["weight_decay"]
    lr_decay = cf["train"]["lr_decay"]

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay) 
    #if USE_CUDA:
    #    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    from visdom import Visdom
    viz = Visdom()
    title = "100h Chinese Corpus with LAS Model"
    opts = [dict(title=title+" Loss on Train", ylabel = 'Train Loss', xlabel = 'Epoch'),
            dict(title=title+" CER on Train", ylabel = 'Train CER', xlabel = 'Epoch'),
            dict(title=title+" Loss on CV", ylabel = 'CV Loss', xlabel = 'Epoch'),
            dict(title=title+' CER on CV', ylabel = 'CV CER', xlabel = 'Epoch')]
    viz_window = [None, None, None, None]
    
    lr = init_lr
    cer_best = 100
    
    start_time = time.time()
    train_loss_results = []
    train_cer_results = []
    cv_loss_results = []
    cv_cer_results = []
    
    for count in range(epochs):
        count += 1
        
        if count in decay_epoch:
            lr *= lr_decay
            for param in optimizer.param_groups:
                param['lr'] *= lr_decay
        
        print("Epoch: %d, learning_rate: %.5f" % (count, lr))
        
        train_loss, train_cer = run_epoch(model, train_loader, optimizer=optimizer, is_training=True, print_every=200, USE_CUDA=USE_CUDA)
        train_loss_results.append(train_loss)
        train_cer_results.append(train_cer)
        cv_loss, cer = run_epoch(model, cv_loader, is_training=False, USE_CUDA=USE_CUDA)
        cv_loss_results.append(cv_loss)
        cv_cer_results.append(cer)
        
        time_used = (time.time() - start_time) / 60
        print("Epoch %d done | TrLoss:%.4 TrCER:%.4f | CvLoss: %.4f CvCER:%.4f | Time_used: %.4f minutes" % (count, train_loss, train_cer, cv_loss, cer, time_used))
        print("Epoch %d done | TrLoss:%.4 TrCER:%.4f | CvLoss: %.4f CvCER:%.4f | Time_used: %.4f minutes" % (count, train_loss, train_cer, cv_loss, cer, time_used), file=log_file)
        
        x_axis = range(count)
        y_axis = [train_loss_results[0:count], train_cer_results[0:count], cv_loss_results[0:count], cv_cer_results[0:count]]
        for x in range(len(viz_window)):
            if viz_window[x] is None:
                viz_window[x] = viz.line(X = np.array(x_axis), Y = np.array(y_axis[x]), opts = opts[x],)
            else:
                viz.line(X = np.array(x_axis), Y = np.array(y_axis[x]), win = viz_window[x], update = 'replace',)

        epoch_path = os.path.join(checkpoints, cf['data']['exp']+'.checkpoint.epoch%d' % count)
        torch.save(LAS.save_package(model, optimizer=optimizer), epoch_path)

        if cer < cer_best:
            best_path = os.path.join(checkpoints, cf['data']['exp']+'best_model.checkpoint')
        torch.save(LAS.save_package(model, optimizer=optimizer), best_path)

if __name__ == '__main__':
    main()



