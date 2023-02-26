import torch
import torch.nn as nn
from modules import Variation, ContextEncoder, MLP
from diffuser import NoiseScheduler
from tqdm.auto import tqdm
from torch.nn import functional as F
import numpy as np



class DiffAE(nn.Model):
    def __init__(self, conf):
        super().___init__()
        # parameters
        self.diff_lr = conf.diff_lr
        self.diff_train_epoch = conf.diff_train_epoch
        # nn
        self.prior_net = Variation(conf['n_hidden'], conf['z_size']) # p(e|x)
        self.prior_generator = nn.Sequential( 
                                nn.Linear(conf['z_size'], conf['z_size']),
                                nn.BatchNorm1d(conf['z_size'], eps=1e-05, momentum=0.1),
                                nn.ReLU(),
                                nn.Linear(conf['z_size'], conf['z_size']),
                                nn.BatchNorm1d(conf['z_size'], eps=1e-05, momentum=0.1),
                                nn.ReLU(),
                                nn.Linear(conf['z_size'], conf['z_size'])
                            ) 
        self.post_net = Variation(conf['n_hidden']*3, conf['z_size']) # q(e|c,x)
        self.diffuser = MLP()
        self.context_encoder = ContextEncoder(
                                    self.utt_encoder, 
                                    conf['n_hidden']*2+2, 
                                    conf['n_hidden'], 1, 
                                    conf['noise_radius']
                                ) 
        self.mlp = nn.Linear()

    def sample_code_prior(self, c):
        e, _, _ = self.prior_net(c)
        z = self.prior_generator(e)
        return z 

    def get_diff_data(self, c):
        eps = sample_code_prior(self, c)



    def train_diffuser(self, dataloader):
        model = self.diffuser
        self.noise_scheduler = NoiseScheduler()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.diff_lr
        )
        global_step = 0
        print("Training diffuser...")
        for epoch in range(self.diff_train_epoch):
            model.train()
            progress_bar = tqdm(total=len(dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(dataloader):
                batch = batch[0]
                noise = torch.randn(batch.shape)
                timesteps = torch.randint(
                    0, self.noise_scheduler.num_timesteps, (batch.shape[0],)
                ).long()
                noisy = self.noise_scheduler.add_noise(batch, noise, timesteps)
                noise_pred = model(noisy, timesteps)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward(loss)

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1
            progress_bar.close()

    def sample_diffuser(self, num_samples):
        model = self.diffuser
        model.eval()
        sample = torch.randn(num_samples, 2)
        timesteps = list(range(len(self.noise_scheduler)))[::-1]
        for i, t in enumerate(tqdm(timesteps)):
            t = torch.from_numpy(np.repeat(t, num_samples)).long()
            with torch.no_grad():
                residual = model(sample, t)
            sample = self.noise_scheduler.step(residual, t[0], sample)
        






