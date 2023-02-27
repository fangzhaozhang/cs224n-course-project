import torch
import torch.nn as nn
from modules import LatentEncoder, ContextEncoder, MLP, Decoder, Encoder, LatentDecoder
from diffuser import NoiseScheduler
from tqdm.auto import tqdm
from torch.nn import functional as F
import numpy as np
import torch.optim as optim



class DiffAE(nn.Model):
    def __init__(self, conf, vocab_size, PAD_token=0):
        super().___init__()
        # parameters
        self.diff_lr = conf.diff_lr
        self.diff_train_epoch = conf.diff_train_epoch
        self.decoder_phase1_train_epoch = conf.decoder_phase1_train_epoch
        self.decoder_phase2_train_epoch = conf.decoder_phase2_train_epoch
        self.vocab_size = vocab_size
        self.temp=conf['temp'] 
        self.clip = conf['clip']
        # nn
        self.embedder= nn.Embedding(vocab_size, 
                                    conf['emb_size'], 
                                    padding_idx=PAD_token)
        self.utt_encoder = Encoder(self.embedder, 
                                   conf['emb_size'], 
                                   conf['n_hidden'], 
                                    True, 
                                   conf['n_layers'], 
                                   conf['noise_radius']) 
        self.context_encoder = ContextEncoder(
                                    self.utt_encoder, 
                                    conf['n_hidden']*2+2, 
                                    conf['n_hidden'], 1, 
                                    conf['noise_radius']) 
        self.prior_net = LatentEncoder(conf['n_hidden'], conf['z_size']) # p(e|x)
        self.post_net = LatentEncoder(conf['n_hidden']*3, conf['z_size']) # q(e|c,x)
        self.diffuser = MLP()
        self.decoder = LatentDecoder() 
        self.respsonse_decoder = Decoder(
                                    self.embedder, 
                                    conf['emb_size'], 
                                    conf['n_hidden']+conf['z_size'], 
                                    vocab_size, 
                                    n_layers=1) 
        self.optimizer_diff = optim.AdamW(self.diffuser.parameters(),
                                        lr=self.diff_lr
                                        )
        self.optimizer_AE_phase1 = optim.SGD(list(self.context_encoder.parameters())
                                +list(self.post_net.parameters()),
                                lr=conf['lr_ae'])
        self.optimizer_AE_phase2 = optim.SGD(list(self.context_encoder.parameters())
                                +list(self.prior_net.parameters())
                                +list(self.decoder.parameters())
                                +list(self.respsonse_decoder.parameters()),
                                lr=conf['lr_ae'])
        self.criterion_ce = nn.CrossEntropyLoss()
        self.noise_scheduler = NoiseScheduler()
        self.lr_scheduler_AE = optim.lr_scheduler.StepLR(self.optimizer_AE, step_size = 10, gamma=0.6)

    def get_diff_data(self, context, response, context_lens, utt_lens, floors, res_lens):
        self.post_net.eval()
        with torch.no_grad():
            x,_ = self.utt_encoder(response[:,1:], res_lens-1)
            c = self.context_encoder(context, context_lens, utt_lens, floors)
            eps, _, _ = self.post_net(torch.cat((x, c),1))
        return eps

    def train_diffuser(self, context, response, context_lens, utt_lens, floors, res_lens):
        print("Training diffuser...")
        self.diffuser.train()
        loss_list = []
        eps = self.get_diff_data(context, response, context_lens, utt_lens, floors, res_lens)
        for _ in enumerate(tqdm(range(self.diff_train_epoch))):
            noise = torch.randn(eps.shape)
            timesteps = torch.randint(
                0, self.noise_scheduler.num_timesteps, (eps.shape[0],)
            ).long()
            noisy = self.noise_scheduler.add_noise(eps, noise, timesteps)
            noise_pred = self.diffuser(noisy, timesteps)
            self.optimizer_diff.zero_grad()
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(self.diffuser.parameters(), 1.0)
            self.optimizer_diff.step()
            loss_list.append(loss.detach().item())
        loss_diff = torch.Tensor(loss_list).mean()
        return loss_diff

    def sample_diffuser(self, num_samples):
        print("Sampling diffuser...")
        self.diffuser.eval()
        sample = torch.randn(num_samples, 2)
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        for _, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, num_samples)).long()
            with torch.no_grad():
                residual = self.diffuser(sample, t)
            sample = self.noise_scheduler.step(residual, t[0], sample)

    def train_decoder_phase1(self, context, response, context_lens, utt_lens, floors, res_lens):
        print("Training AE phase 1...")
        self.context_encoder.train()
        self.post_net.train()
        self.prior_net.eval()
        self.decoder.eval()
        self.respsonse_decoder.eval()
        loss_list = []
        for _ in enumerate(tqdm(range(self.decoder_phase1_train_epoch))):
            x,_ = self.utt_encoder(response[:,1:], res_lens-1)
            c = self.context_encoder(context, context_lens, utt_lens, floors)
            e_prior, _, _ = self.prior_net(c)
            e_post, _, _ = self.post_net(torch.cat((x, c),1))
            decoder_out = self.decoder(torch.cat((e_prior, e_post),1))
            output = self.respsonse_decoder(torch.cat((decoder_out, c),1))
            flattened_output = output.view(-1, self.vocab_size) 
            dec_target = response[:,1:].contiguous().view(-1)
            mask = dec_target.gt(0)
            masked_target = dec_target.masked_select(mask)
            output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)
            masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
            self.optimizer_AE_phase1.zero_grad()
            loss = self.criterion_ce(masked_output/self.temp, masked_target)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.context_encoder.parameters())+list(self.decoder.parameters()), 
                self.clip)
            self.optimizer_AE_phase1.step()
            loss_list.append(loss.detach().item())
        loss_AE1 = torch.Tensor(loss_list).mean()
        return loss_AE1


    def train_decoder_phase2(self, context, response, context_lens, utt_lens, floors):
        print("Training AE phase 2...")
        self.context_encoder.train()
        self.diffuser.eval()
        self.prior_net.train()
        self.decoder.train()
        self.respsonse_decoder.train()
        loss_list = []
        with torch.no_grad():
            e_post = self.sample_diffuser(self, len(context))
        for _ in enumerate(tqdm(range(self.decoder_phase1_train_epoch))):
            c = self.context_encoder(context, context_lens, utt_lens, floors)
            e_prior, _, _ = self.prior_net(c)
            decoder_out = self.decoder(torch.cat((e_prior, e_post),1))
            output = self.respsonse_decoder(torch.cat((decoder_out, c),1))  
            flattened_output = output.view(-1, self.vocab_size) 
            dec_target = response[:,1:].contiguous().view(-1)
            mask = dec_target.gt(0)
            masked_target = dec_target.masked_select(mask)
            output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)
            masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
            self.optimizer_AE_phase2.zero_grad()
            loss = self.criterion_ce(masked_output/self.temp, masked_target)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.context_encoder.parameters())+list(self.decoder.parameters()), 
                self.clip)
            self.optimizer_AE_phase2.step()
            loss_list.append(loss.detach().item())
        loss_AE2 = torch.Tensor(loss_list).mean()
        return loss_AE2
    
    def adjust_lr(self):
        self.lr_scheduler_AE.step()

    def valid(self, context, response, context_lens, utt_lens, floors):
        self.context_encoder.eval()
        self.prior_net.eval()
        self.post_net.eval()
        self.diffuser.eval()
        self.decoder.eval()
        self.respsonse_decoder.eval()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        e_post = self.sample_diffuser(self, len(context))
        e_prior, _, _ = self.prior_net(c)
        decoder_out = self.decoder(torch.cat((e_prior, e_post),1))
        output = self.respsonse_decoder(torch.cat((decoder_out, c),1))
        flattened_output = output.view(-1, self.vocab_size) 
        dec_target = response[:,1:].contiguous().view(-1)
        mask = dec_target.gt(0)
        masked_target = dec_target.masked_select(mask)
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        with torch.no_grad():
            loss = self.criterion_ce(masked_output/self.temp, masked_target)
        return loss.detach().item()
    
    def sample(self, context, context_lens, utt_lens, floors, repeat, SOS_tok, EOS_tok):    
        self.context_encoder.eval()
        self.prior_net.eval()
        self.post_net.eval()
        self.diffuser.eval()
        self.decoder.eval()
        self.respsonse_decoder.eval()

        c = self.context_encoder(context, context_lens, utt_lens, floors)
        e_post = self.sample_diffuser(self, len(context))
        e_prior, _, _ = self.prior_net(c)
        decoder_out = self.decoder(torch.cat((e_prior, e_post),1))
        sample_words, sample_lens= self.respsonse_decoder.sampling(torch.cat((decoder_out, c),1), 
                                                         None, self.maxlen, SOS_tok, EOS_tok, "greedy") 
        return sample_words, sample_lens 



        





