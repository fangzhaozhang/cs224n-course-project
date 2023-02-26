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

parser = argparse.ArgumentParser(description='DialogDiffAE')
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument("--output_file", type=str, default="out")
parser.add_argument('--dataset', type=str, default='DailyDial')
parser.add_argument('--valid_every', type=int, default=500, help='interval to validation')
parser.add_argument('--eval_every', type=int, default=2, help='interval to evaluate on the validation set')

args = parser.parse_args()
outdir = f"exps/{args.output_file}"
os.makedirs(outdir, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conf = {
    'emb_size': 200,
    'maxlen': 40,
    'epochs': 100,
    'batch_size': 32,
    'diaglen': 10,
    'n_hidden': 300,
    'z_size': 200,
    'noise_radius': 0.2,
    'diff_lr': 1e-3,
    'diff_train_epoch': 50
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
# ToDo
model = model.DiffAE().to(device)
print("Loaded word2vec as model embedder")
model.embedder.weight.data.copy_(torch.from_numpy(corpus.word2vec))
model.embedder.weight.data[0].fill_(0)

"""
Train Model
"""
print("Training...")

for epoch in range(conf['epochs']+1):
    train_loader.epoch_init(conf['batch_size'], conf['diaglen'], 1, shuffle=True)
    model.train()
    progress_bar = tqdm(total=len(train_loader))
    progress_bar.set_description(f"Epoch {epoch}")

    global_step = 1
    for step, batch in enumerate(train_loader):
        batch = batch[0]
        # ToDo
        loss_AE = model.train_AE()
        loss_Diff = model.train_Diff()
        progress_bar.update(1)
        logs = {
                "(loss_AE, loss_Diff):": (loss_AE.detach().item(), loss_Diff.detach().item()),
                "step": global_step
                }
        progress_bar.set_postfix(**logs)
        global_step += 1
    progress_bar.close()


    """
    Validate Model 
    """
    if epoch % args.eval_every == 0:
        print("Validating...")
        valid_loader.epoch_init(1, conf['diaglen'], 1, shuffle=False)
        model.eval()
        # ToDo
        evaluate(model, metrics, valid_loader)

    model.adjust_lr()

"""
Test Model 
"""
print("Testing...")
test_loader.epoch_init(1, conf['diaglen'], 1, shuffle=False)
model.eval()
evaluate(model, metrics, test_loader)




    





# """
# Validate Model after training on #args.valid_ever data batches
# """
# if global_step % args.valid_every == 0:
#     valid_loader.epoch_init(conf['batch_size'], conf['diaglen'], 1, shuffle=False)
#     model.eval()
#     progress_bar = tqdm(total=len(valid_loader))
#     progress_bar.set_description(f"Validate at Epoch {epoch}")
#     for step, batch in enumerate(valid_loader):
#         batch = batch[0]
#         # ToDo
#         valid_loss = model.valid()
#         progress_bar.update(1)
#         progress_bar.set_postfix(
#             f"validation loss: {valid_loss}"
#         )
#     progress_bar.close()
















