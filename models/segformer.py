import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
import pdb
# from torchinfo import summary
from torch import optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from torchmetrics import JaccardIndex
from datasets import load_metric
from transformers import SegformerForSemanticSegmentation

#referenced from https://blog.roboflow.com/how-to-train-segformer-on-a-custom-dataset-with-pytorch-lightning/

class LapNetSegformer(pl.LightningModule):

    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear,
        learning_rate,
        epsilon,
    ):
        super(LapNetSegformer, self).__init__()

        self.n_channels = n_channels
        self.num_classes = n_classes
        self.bilinear = bilinear
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # Metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", 
            return_dict=False, 
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True,
        )

        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")

    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return(outputs)

    def _shared_step(self, batch, batch_idx):
        images, masks = batch

        outputs = self(images, masks.long()) 
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)


        return (loss, predicted, images, masks, outputs)
    def training_step(self, batch, batch_idx):
        loss, predicted, images, masks, outputs = self._shared_step(batch, batch_idx)

        self.train_mean_iou.add_batch(
                    predictions=predicted.detach().cpu().numpy(), 
                    references=masks.detach().cpu().numpy()
        )

        metrics = self.train_mean_iou.compute(
            num_labels=self.num_classes, 
            ignore_index=255, 
            reduce_labels=False,
        )

        metrics = {'loss': loss, 'train/loss': loss, "train_mean_iou": metrics["mean_iou"], "train_mean_accuracy": metrics["mean_accuracy"]}
            
        for k,v in metrics.items():
            self.log(k,v)
            
        return(metrics)

    def validation_step(self, batch, batch_idx):
        loss, predicted, images, masks, outputs = self._shared_step(batch, batch_idx)

        #self.log('val/loss', loss)

        #print('val/loss', loss)

        self.val_mean_iou.add_batch(
                    predictions=predicted.detach().cpu().numpy(), 
                    references=masks.detach().cpu().numpy()
                )

        metrics = self.val_mean_iou.compute(
            num_labels=self.num_classes, 
            ignore_index=255, 
            reduce_labels=False,
        )

        metrics = {'loss': loss, 'val/loss': loss, "val_mean_iou": metrics["mean_iou"], "val_mean_accuracy": metrics["mean_accuracy"]}
            
        for k,v in metrics.items():
            self.log(k,v)
            
        return(metrics)

        '''
        #create a dictionary mapping indices from 0 to 13 to different random colors

        colors_dict = {0: (0,0,0), 1: (255,0,0), 2: (0,255,0), 3: (0,0,255), 4: (255,255,0), 5: (255,0,255), 6: (0,255,255), 7: (255,255,255), 8: (128,0,0), 9: (0,128,0), 10: (0,0,128), 11: (128,128,0), 12: (128,0,128), 13: (0,128,128)}

        #0: black, 1: red, 2: green, 3: blue, 4: yellow, 5: magenta, 6: cyan, 7: white, 8: maroon, 9: olive, 10: navy, 11: purple, 12: teal, 13: silver
        
        label = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        colors = ['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white', 'maroon', 'olive', 'navy', 'purple', 'teal', 'silver']

        

        for ind, img in enumerate(input_images):

            plt.imshow(img.cpu().detach().numpy().transpose((1,2,0)))# cmap=matplotlib.colors.ListedColormap(colors))
            plt.axis('off')
            plt.savefig('val_images/input/input_image_{}.png'.format(ind))


        for ind, img in enumerate(predicted_images):

            plt.imshow(img.cpu().detach().numpy(), cmap=matplotlib.colors.ListedColormap(colors))
            plt.axis('off')
            plt.savefig('val_images/pred/val_image_{}.png'.format(ind))

        for ind, img in enumerate(target_images):

            plt.imshow(img.cpu().detach().numpy(), cmap=matplotlib.colors.ListedColormap(colors))
            plt.axis('off')
            plt.savefig('val_images/target/target_image_{}.png'.format(ind))
            #pdb.set_trace()

        '''
        
        return losses['loss']
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, eps=self.epsilon)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_steps)
        
        return [optimizer], [lr_scheduler]


if __name__ == '__main__':
    pass