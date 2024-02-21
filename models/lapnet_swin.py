import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
import pdb
import json

from transformers import SwinModel
from transformers.utils import logging
logging.set_verbosity(50) # Silence warning when loading pretrained weights/bias without last layer
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class MLP(nn.Module):
    """
    A simple MLP with 2 hidden layers. Used for the auxiliary tasks.
    """
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, query_dim=512, key_dim=512, value_dim=512, num_heads=8):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        
        self.Wq = nn.Linear(query_dim, query_dim * num_heads, bias=False)
        self.Wk = nn.Linear(key_dim, key_dim * num_heads, bias=False)
        self.Wv = nn.Linear(value_dim, value_dim * num_heads, bias=False)
        self.Wo = nn.Linear(value_dim * num_heads, value_dim)

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, queries, keys, values):
        Q = self.Wq(queries)
        K = self.Wk(keys)
        V = self.Wv(values)
        
        Q = Q.view(-1, self.num_heads, self.query_dim)
        K = K.view(-1, self.num_heads, self.key_dim)
        V = V.view(-1, self.num_heads, self.value_dim)
        
        attn = torch.bmm(Q, K.transpose(1, 2))
        attn = self.softmax(attn)
        
        out = torch.bmm(attn, V)
        out = out.view(-1, self.num_heads * self.value_dim)
        out = self.Wo(out)
        
        return out
    

class LapNetSwin(pl.LightningModule):
    """
    LapNetSwin model using Swin Transformer for the encoder of the temporal branch.
    """
    def __init__(
        self, 
        run_name, 
        drop_path_rate, 
        blocks_freeze, 
        learning_rate, 
        epsilon,
        num_subjects, 
        num_verbs, 
        num_targets,
        log_cfms,
        lambda_subject,
        lambda_verb,
        lambda_target,
    ):
        super().__init__()

        self.run_name = run_name
        self.drop_path_rate = drop_path_rate
        self.blocks_freeze = blocks_freeze
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_subjects = num_subjects
        self.num_verbs = num_verbs
        self.num_targets = num_targets
        self.log_cfms = log_cfms
        self.lambda_subject = lambda_subject
        self.lambda_verb = lambda_verb
        self.lambda_target = lambda_target

        # Load pretrained Swin transformer
        model_kwargs = dict(drop_path_rate = self.drop_path_rate)
        self.encoder = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224", **model_kwargs)
        
        # Freeze certain blocks of encoder
        print(f"Number of trainable Swin stages = {self.encoder.num_layers}/4")
        ct = 0
        for child in self.encoder.children():
            if ct < self.blocks_freeze:
                for param in child.parameters():
                    param.requires_grad = False
                # print('freeze layer: ', ct)
            else:
                for param in child.parameters():
                    param.requires_grad = True
                # print('unfreezing layer: ', ct)
            ct += 1
        
        self.subject_head = MLP(768, self.num_subjects)
        self.verb_head = MLP(768, self.num_verbs)
        self.target_head = MLP(768, self.num_targets)

        # self.verb_cross_attn = CrossAttention(768, self.num_subjects, self.num_subjects)
        # self.target_cross_attn = CrossAttention(768, self.num_verbs, self.num_verbs)

        # Metrics
        self.train_subject_acc = torchmetrics.Accuracy()
        self.train_verb_acc = torchmetrics.Accuracy()
        self.train_target_acc = torchmetrics.Accuracy()
        self.train_mean_acc = torchmetrics.MeanMetric()

        self.val_subject_acc = torchmetrics.Accuracy()
        self.val_verb_acc = torchmetrics.Accuracy()
        self.val_target_acc = torchmetrics.Accuracy()
        self.val_mean_acc = torchmetrics.MeanMetric()

        self.test_subject_acc = torchmetrics.Accuracy()
        self.test_verb_acc = torchmetrics.Accuracy()
        self.test_target_acc = torchmetrics.Accuracy()
        self.test_mean_acc = torchmetrics.MeanMetric()

    def forward(self, x):
        x = x.half()
        z = self.encoder(x).pooler_output # pooler logits

        out_s = self.subject_head(z)
        
        # Cross attention between subject and verb
        # z = self.verb_cross_attn(z, out_s, out_s)
        out_v = self.verb_head(z)

        # Cross attention between verb and target
        # z = self.target_cross_attn(z, out_v, out_v)
        out_t = self.target_head(z)
        
        return out_s, out_v, out_t

    def _shared_step(self, batch, batch_idx):
        # Batch contains ground truth labels for subject, verb and target
        x, (y_subject, y_verb, y_target) = batch

        # Model output logits for subject, verb and target
        y_hat_subject, y_hat_verb, y_hat_target = self(x)

        # Calculate individual loss for each task
        subject_loss = F.cross_entropy(y_hat_subject, y_subject)
        verb_loss = F.cross_entropy(y_hat_verb, y_verb)
        target_loss = F.cross_entropy(y_hat_target, y_target)

        # Different weightings can be used for each task 
        # (to simulate single task learning, we'll set the weights of other losses to 0)
        ls, lv, lt = self.lambda_subject, self.lambda_verb, self.lambda_target

        # Aggregate losses
        loss = ls*subject_loss + lv*verb_loss + lt*target_loss

        return (
            {'loss': loss, 'subject_loss': subject_loss, 'verb_loss': verb_loss, 'target_loss': target_loss}, 
            {'yhat_s': y_hat_subject, 'yhat_v': y_hat_verb, 'yhat_t': y_hat_target}, 
            {'y_s': y_subject, 'y_v': y_verb, 'y_t': y_target},
        )

    def training_step(self, batch, batch_idx):
        losses, preds, gt = self._shared_step(batch, batch_idx)

        self.log('train/loss', losses['loss'])

        self.log('train/subject_loss', losses['subject_loss'], on_step=True, on_epoch=False)
        self.log('train/verb_loss', losses['verb_loss'], on_step=True, on_epoch=False)
        self.log('train/target_loss', losses['target_loss'], on_step=True, on_epoch=False)

        self.train_subject_acc(preds['yhat_s'], gt['y_s'])
        self.log('train_acc_subject', self.train_subject_acc, on_step=False, on_epoch=True)

        self.train_verb_acc(preds['yhat_v'], gt['y_v'])
        self.log('train_acc_verb', self.train_verb_acc, on_step=False, on_epoch=True)

        self.train_target_acc(preds['yhat_t'], gt['y_t'])
        self.log('train_acc_target', self.train_target_acc, on_step=False, on_epoch=True)
  
        # calculate mean accuracy and log
        self.train_mean_acc(torch.stack([
            self.train_subject_acc.compute(),
            self.train_verb_acc.compute(),
            self.train_target_acc.compute()
        ]).mean())

        self.log('train_mean_acc', self.train_mean_acc, on_step=False, on_epoch=True)

        return losses['loss']

    def validation_step(self, batch, batch_idx):
        losses, preds, gt = self._shared_step(batch, batch_idx)

        self.log('val/loss', losses['loss'])

        self.log('val/subject_loss', losses['subject_loss'], on_step=True, on_epoch=False)
        self.log('val/verb_loss', losses['verb_loss'], on_step=True, on_epoch=False)
        self.log('val/target_loss', losses['target_loss'], on_step=True, on_epoch=False)

        self.val_subject_acc(preds['yhat_s'], gt['y_s'])
        self.log('val_acc_subject', self.val_subject_acc, on_step=False, on_epoch=True)

        self.val_verb_acc(preds['yhat_v'], gt['y_v'])
        self.log('val_acc_verb', self.val_verb_acc, on_step=False, on_epoch=True)

        self.val_target_acc(preds['yhat_t'], gt['y_t'])
        self.log('val_acc_target', self.val_target_acc, on_step=False, on_epoch=True)

        self.val_mean_acc(torch.stack([
            self.val_subject_acc.compute(),
            self.val_verb_acc.compute(),
            self.val_target_acc.compute()
        ]).mean())

        self.log('val_mean_acc', self.val_mean_acc, on_step=False, on_epoch=True)

        return preds, gt
    
    def test_step(self, batch, batch_idx):
        losses, preds, gt = self._shared_step(batch, batch_idx)

        self.log('test/loss', losses['loss'])

        self.log('test/subject_loss', losses['subject_loss'], on_step=True, on_epoch=False)
        self.log('test/verb_loss', losses['verb_loss'], on_step=True, on_epoch=False)
        self.log('test/target_loss', losses['target_loss'], on_step=True, on_epoch=False)

        self.test_subject_acc(preds['yhat_s'], gt['y_s'])
        self.log('test_acc_subject', self.test_subject_acc, on_step=False, on_epoch=True)

        self.test_verb_acc(preds['yhat_v'], gt['y_v'])
        self.log('test_acc_verb', self.test_verb_acc, on_step=False, on_epoch=True)

        self.test_target_acc(preds['yhat_t'], gt['y_t'])
        self.log('test_acc_target', self.test_target_acc, on_step=False, on_epoch=True)

        self.test_mean_acc(torch.stack([
            self.test_subject_acc.compute(),
            self.test_verb_acc.compute(),
            self.test_target_acc.compute()
        ]).mean())

        self.log('test_mean_acc', self.test_mean_acc, on_step=False, on_epoch=True)

        return preds, gt
    
    
    def validation_epoch_end(self, outs):
        sub_preds = torch.hstack([out[0]['yhat_s'].softmax(dim=1).argmax(dim=1) for out in outs])
        verb_preds = torch.hstack([out[0]['yhat_v'].softmax(dim=1).argmax(dim=1) for out in outs])
        targ_preds = torch.hstack([out[0]['yhat_t'].softmax(dim=1).argmax(dim=1) for out in outs])

        # Make triplet classes from predictions
        preds = torch.stack([sub_preds, verb_preds, targ_preds], dim=1)
        
        # Make new class for each triplet
        preds = preds[:, 0] * self.num_verbs * self.num_targets + preds[:, 1] * self.num_targets + preds[:, 2]
        
        sub_gt = torch.hstack([out[1]['y_s'] for out in outs])
        verb_gt = torch.hstack([out[1]['y_v'] for out in outs])
        targ_gt = torch.hstack([out[1]['y_t'] for out in outs])

        # Make triplet classes from ground truth
        gt = torch.stack([sub_gt, verb_gt, targ_gt], dim=1)

        # Make new class for each triplet
        gt = gt[:, 0] * self.num_verbs * self.num_targets + gt[:, 1] * self.num_targets + gt[:, 2]

        if self.log_cfms:
            print('Logging confusion matrices ...')
            self.log_confusion_matrix(sub_preds, sub_gt, self.num_subjects, 'subject')
            self.log_confusion_matrix(verb_preds, verb_gt, self.num_verbs, 'verb')
            self.log_confusion_matrix(targ_preds, targ_gt, self.num_targets, 'target')

            self.log_confusion_matrix(preds, gt, self.num_subjects * self.num_verbs * self.num_targets, 'triplet')

        # Cleanup
        del sub_preds, verb_preds, targ_preds, preds, sub_gt, verb_gt, targ_gt, gt

    def log_confusion_matrix(self, preds, gt, num_classes, name):
        if not os.path.exists(f"metrics"):
            os.mkdir(f"metrics")
            os.mkdir(f"metrics/cfms")

        matrix = np.nan_to_num(confusion_matrix(gt.cpu(), preds.cpu(), labels=range(num_classes)))

        with open(f'metrics/{self.run_name}_{name}_classwise_val_acc.pkl', 'wb') as f:
            pickle.dump(matrix.diagonal()/(matrix.sum(axis=1) + 1e-8), f)

        if name == 'triplet':
            return
            
        with open(f"metrics/cfms/{self.run_name}_{name}_cfm", 'w') as f:
            f.write("\n\nRun: "+self.run_name+"\n")
            f.write(np.array2string(matrix, separator=', '))

        cmn = matrix.astype('float') / (matrix.sum(axis=1)[:, np.newaxis]+1e-8)

        names = ...
        with open(f'data/{name}s.jsonl', 'r') as f:
            names = json.load(f)

        xticks = [names[str(i)] for i in range(matrix.shape[0])]
        yticks = [names[str(i)] for i in range(matrix.shape[0])]

        fig, _ = plt.subplots(figsize=(12, 9))
        cm_plot = sns.heatmap(
            cmn, annot=True, fmt='.2f', 
            xticklabels=xticks, yticklabels=yticks,
            cmap = "rocket"
        )
        cm_plot.set_xticklabels(cm_plot.get_xticklabels(), rotation=90)
        plt.xlabel(f'Predicted {name}')
        plt.ylabel(f'Actual {name}')

        title = ''
        if name == 'subject':
            title = 'Instrument'
        elif name == 'verb':
            title = 'Verb'
        elif name == 'target':
            title = 'Target'

        plt.title(title)
        plt.tight_layout()
        plt.show(block=False)

        fig = cm_plot.get_figure()
        fig.savefig(f"metrics/cfms/{self.run_name}_{name}_cfm.png", dpi=300)

        # Cleanup
        del matrix
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, eps=self.epsilon)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_steps)
        
        return [optimizer], [lr_scheduler]


if __name__ == '__main__':
    pass