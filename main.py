"""
Copyright 2018 NAVER Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Adapted By Fangzhao, Xiaohan
"""

import torch
import argparse
import data
import os
import model
import evaluate
from metrics import Metrics
from tqdm.auto import tqdm
from helper import gVar, gData

parser = argparse.ArgumentParser(description='DialogDiffAE')
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument("--output_file", type=str, default="out")
parser.add_argument('--dataset', type=str, default='DailyDial')
parser.add_argument('--valid_every', type=int, default=500, help='interval to validation')
parser.add_argument('--eval_every', type=int, default=2, help='interval to evaluate on the validation set')
parser.add_argument('--n_samples', type=int, default=10, help='Number of responses to sampling')

args = parser.parse_args()
outdir = f"exps/{args.output_file}"
os.makedirs(outdir, exist_ok=True)
 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = {
    'emb_size': 200,
    'maxlen': 40,
    'global_epochs': 100,
    'batch_size': 32,
    'diaglen': 10,
    'n_hidden': 300,
    'z_size': 200,
    'noise_radius': 0.2,
    'diff_lr': 1e-3,
    'diff_train_epoch': 10,
    'decoder_phase1_train_epoch': 10,
    'decoder_phase2_train_epoch': 10,
    'ae_lr': 1,
    'temp': 1,
    'clip': 1
}

"""
Load Data
"""
data_path=args.data_path+args.dataset+'/'
corpus = getattr(data, args.dataset+'Corpus')(
                    data_path, 
                    wordvec_path=args.data_path+'glove.twitter.27B.200d.txt', 
                    wordvec_dim=conf['emb_size']
                )
dials = corpus.get_dialogs()
metas = corpus.get_metas()
train_dial = dials.get("train")
valid_dial = dials.get("valid")
test_dial = dials.get("test")
train_meta = metas.get("train") 
valid_meta = metas.get("valid")
test_meta = metas.get("test")
train_loader = getattr(data, args.dataset+'DataLoader')(
                    "Train", 
                    train_dial, 
                    train_meta, 
                    conf['maxlen']
                )
valid_loader = getattr(data, args.dataset+'DataLoader')(
                    "Valid", 
                    valid_dial, 
                    valid_meta, 
                    conf['maxlen']
                )
test_loader = getattr(data, args.dataset+'DataLoader')(
                    "Test", 
                    test_dial, 
                    test_meta, 
                    conf['maxlen']
                )
vocab = corpus.ivocab
ivocab = corpus.vocab
n_tokens = len(ivocab)
metrics = Metrics(corpus.word2vec)

print("Loaded data!")


"""
Build Model 
"""
model = model.DiffAE(conf, n_tokens).to(device)
print("Loaded word2vec as model embedder")
model.embedder.weight.data.copy_(torch.from_numpy(corpus.word2vec))
model.embedder.weight.data[0].fill_(0)

"""
Train Model
"""
print("Training Global Model ...")

for epoch in range(conf['global_epochs']):
    train_loader.epoch_init(conf['batch_size'], conf['diaglen'], 1, shuffle=True)
    model.train()
    progress_bar = tqdm(total=len(train_loader))
    progress_bar.set_description(f"Global Epoch {epoch}")
    global_step = 0
    for step, batch in enumerate(train_loader):
        batch = batch[0]
        context,context_lens,utt_lens,floors,_,_,_,response,res_lens,_ = batch
        context,utt_lens = context[:,:,1:], utt_lens-1
        context, context_lens, utt_lens, floors, response, res_lens\
                = gVar(context), gVar(context_lens), gVar(utt_lens), gData(floors), gVar(response), gVar(res_lens)
        ######### TODO #########
        loss_AE_phase1 = model.train_AE_phase1(context, response, context_lens, utt_lens, floors, res_lens)
        ######### TODO #########
        loss_diff = model.train_diffuser(context, response, context_lens, utt_lens, floors, res_lens)
        ######### TODO #########
        loss_AE_phase2 = model.train_AE_phase2(context, response, context_lens, utt_lens, floors)
        progress_bar.update(1)
        logs = {
                "(loss_AE1, loss_diff,loss_AE2):": (
                    loss_AE_phase1.detach().item(), 
                    loss_diff.detach().item(),
                    loss_AE_phase2.detach().item()
                    ),
                "step": global_step
                }
        progress_bar.set_postfix(**logs)
        global_step += 1
        """ 
        Validate Model after training on #args.valid_every data batches
        """
        print(f"Validate Model after training on #{args.valid_every} data batches")
        if global_step % args.valid_every == 0:
            valid_loader.epoch_init(conf['batch_size'], conf['diaglen'], 1, shuffle=False)
            model.eval()
            loss_list = []
            for _, batch in enumerate(tqdm(valid_loader)):
                context, context_lens, utt_lens, floors,_,_,_,response,res_lens,_ = batch
                context, utt_lens = context[:,:,1:], utt_lens-1
                context, context_lens, utt_lens, floors, response, res_lens\
                        = gVar(context), gVar(context_lens), gVar(utt_lens), gData(floors), gVar(response), gVar(res_lens)
                ######### TODO ######### 
                valid_loss = model.valid(context, context_lens, utt_lens, floors, response, res_lens)
                loss_list.append(valid_loss)
            loss = torch.Tensor(loss_list).mean()
            print(f"Model validate loss: {loss}")
                    
    progress_bar.close()


    """
    Validate Model 
    """
    if epoch % args.eval_every == 0:
        print("Validating Global Model...")
        valid_loader.epoch_init(1, conf['diaglen'], 1, shuffle=False)
        model.eval()
        repeat = args.n_samples
        ######### TODO ######### 
        recall_bleu, prec_bleu, bow_extrema, bow_avg, bow_greedy, intra_dist1, intra_dist2, avg_len, inter_dist1, inter_dist2\
        =evaluate(model, metrics, valid_loader, vocab, ivocab, None, repeat)                

    ######### TODO ######### 
    model.adjust_lr()

"""
Test Model 
"""
print("Testing...")
test_loader.epoch_init(1, conf['diaglen'], 1, shuffle=False)
model.eval()
repeat = args.n_samples
evaluate(model, metrics, test_loader, vocab, ivocab, None, repeat)


