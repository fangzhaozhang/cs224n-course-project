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
"""

import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import json

import os, sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import indexes2sent, gVar, gData

PAD_token = 0

def evaluate(model, metrics, test_loader, vocab, ivocab, f_eval, repeat):
    
    recall_bleus, prec_bleus, bows_extrema, bows_avg, bows_greedy, intra_dist1s, intra_dist2s, avg_lens, inter_dist1s, inter_dist2s\
        = [], [], [], [], [], [], [], [], [], []
    local_t = 0
    while True:
        batch = test_loader.next_batch()
        if batch is None:
            break
        local_t += 1 
        context, context_lens, utt_lens, floors,_,_,_,response,res_lens,_ = batch   
        context, utt_lens = context[:,:,1:], utt_lens-1 # remove the sos token in the context and reduce the context length
        f_eval.write("Batch %d \n" % (local_t))# print the context
        start = np.maximum(0, context_lens[0]-5)
        for t_id in range(start, context.shape[1], 1):
            context_str = indexes2sent(context[0, t_id], vocab, vocab["</s>"], PAD_token)
            f_eval.write("Context %d-%d: %s\n" % (t_id, floors[0, t_id], context_str))
        # print the true outputs    
        ref_str, _ =indexes2sent(response[0], vocab, vocab["</s>"], vocab["<s>"])
        ref_tokens = ref_str.split(' ')
        f_eval.write("Target >> %s\n" % (ref_str.replace(" ' ", "'")))
        
        context, context_lens, utt_lens, floors = gVar(context), gVar(context_lens), gVar(utt_lens), gData(floors)
        sample_words, sample_lens = model.sample(context, context_lens, utt_lens, floors, repeat, vocab["<s>"], vocab["</s>"])
        # nparray: [repeat x seq_len]
        pred_sents, _ = indexes2sent(sample_words, vocab, vocab["</s>"], PAD_token)
        pred_tokens = [sent.split(' ') for sent in pred_sents]
        for r_id, pred_sent in enumerate(pred_sents):
            f_eval.write("Sample %d >> %s\n" % (r_id, pred_sent.replace(" ' ", "'")))
        max_bleu, avg_bleu = metrics.sim_bleu(pred_tokens, ref_tokens)
        recall_bleus.append(max_bleu)
        prec_bleus.append(avg_bleu)
        
        bow_extrema, bow_avg, bow_greedy = metrics.sim_bow(sample_words, sample_lens, response[:,1:], res_lens-2)
        bows_extrema.append(bow_extrema)
        bows_avg.append(bow_avg)
        bows_greedy.append(bow_greedy)
        
        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = metrics.div_distinct(sample_words, sample_lens)
        intra_dist1s.append(intra_dist1)
        intra_dist2s.append(intra_dist2)
        avg_lens.append(np.mean(sample_lens))
        inter_dist1s.append(inter_dist1)
        inter_dist2s.append(inter_dist2)
                
        f_eval.write("\n")

    recall_bleu = float(np.mean(recall_bleus))
    prec_bleu = float(np.mean(prec_bleus))
    f1 = 2*(prec_bleu*recall_bleu) / (prec_bleu+recall_bleu+10e-12)
    bow_extrema = float(np.mean(bows_extrema))
    bow_avg = float(np.mean(bows_avg))
    bow_greedy=float(np.mean(bows_greedy))
    intra_dist1=float(np.mean(intra_dist1s))
    intra_dist2=float(np.mean(intra_dist2s))
    avg_len=float(np.mean(avg_lens))
    inter_dist1=float(np.mean(inter_dist1s))
    inter_dist2=float(np.mean(inter_dist2s))
    report = "Avg recall BLEU %f, avg precision BLEU %f, F1 %f, bow_extrema %f, bow_avg %f, bow_greedy %f,\
    intra_dist1 %f, intra_dist2 %f, avg_len %f, inter_dist1 %f, inter_dist2 %f (only 1 ref, not final results)" \
    % (recall_bleu, prec_bleu, f1, bow_extrema, bow_avg, bow_greedy, intra_dist1, intra_dist2, avg_len, inter_dist1, inter_dist2)
    print(report)
    f_eval.write(report + "\n")
    print("Done testing")
    
    return recall_bleu, prec_bleu, bow_extrema, bow_avg, bow_greedy, intra_dist1, intra_dist2, avg_len, inter_dist1, inter_dist2