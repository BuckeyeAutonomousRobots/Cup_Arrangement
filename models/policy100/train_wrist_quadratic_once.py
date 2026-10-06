"""ONE 2000-update uniform-sampling candidate; added wrist quadratic loss only."""
import json,os
from pathlib import Path
import torch
import continue_act100 as c
from train_compare import ACTPolicy,IMAGE

class WristQuadraticACT(ACTPolicy):
    def forward(self,batch):
        model_batch=dict(batch)
        model_batch['observation.images']=[batch[key] for key in self.config.image_features]
        pred,(mu,lv)=self.model(model_batch)
        valid=~batch['action_is_pad'].unsqueeze(-1)
        error=pred-batch['action']
        l1=(error.abs()*valid).mean()
        # Preserve all existing L1 terms (including stationary wrist); emphasize large wrist errors.
        extra=.5*(error[:,:,5].square()*valid[:,:,0]).mean()/error.shape[-1]
        kl=(-.5*(1+lv-mu.square()-lv.exp())).sum(-1).mean()
        loss=l1+self.config.kl_weight*kl+extra
        return loss,{'l1_loss':l1.item(),'kld_loss':kl.item(),'wrist_quadratic_extra':extra.item()}

if __name__=='__main__':
    c.OUT=c.t.ROOT/'models/policy100/runs/ACT_wrist_quadratic_20261005'
    c.TARGET_STEPS=7000
    with c.OUT.with_suffix('.claim').open('x') as f:
        json.dump({'pid':os.getpid(),'source_step':5000,'target_step':7000,
            'loss':'original normalized masked L1 + KL10 + 0.5 * masked wrist normalized squared error / 7',
            'sampling':'unchanged uniform shuffled windows','offline_only':True,
            'script_sha256':c.t.sha(Path(__file__))},f);f.flush();os.fsync(f.fileno())
    c.t.ACTPolicy=WristQuadraticACT
    c.main()
