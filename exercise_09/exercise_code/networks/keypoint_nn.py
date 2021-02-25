"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import os
import torchvision.transforms as transforms
import numpy as np
from exercise_code.data.facial_keypoints_dataset import FacialKeypointsDataset

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.Dropout(self.hparams["dropout_p"]),
            nn.ReLU(),
            nn.Linear(256, 30)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def training_step(self, batch, batch_idx):
        images, keypoints = batch['image'], batch['keypoints']

        # forward pass
        predicted = self.forward(images)
        predicted = predicted.view(-1, 15, 2)

        # loss
        criterion = torch.nn.MSELoss(reduction='mean')
        loss = criterion(torch.squeeze(keypoints), torch.squeeze(predicted))

        preds = predicted.argmax()
        n_correct = (keypoints == preds).sum()

        tensorboard_logs = {'loss': loss}

        return {'loss': loss, 'train_n_correct': n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, keypoints = batch['image'], batch['keypoints']

        # forward pass
        predicted = self.forward(images)
        predicted = predicted.view(-1, 15, 2)

        # loss
        criterion = torch.nn.MSELoss(reduction='mean')
        loss = criterion(torch.squeeze(keypoints), torch.squeeze(predicted))

        preds = predicted.argmax()
        n_correct = (keypoints == preds).sum()

        return {'val_loss': loss, 'val_n_correct': n_correct}

    def test_step(self, batch, batch_idx):
        images, keypoints = batch['image'], batch['keypoints']

        # forward pass
        predicted = self.forward(images)
        predicted = predicted.view(-1, 15, 2)

        # loss
        criterion = torch.nn.MSELoss(reduction='mean')
        loss = criterion(torch.squeeze(keypoints), torch.squeeze(predicted))

        preds = predicted.argmax()
        n_correct = (keypoints == preds).sum()

        return {'test_loss': loss, 'test_n_correct': n_correct}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x['val_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.val_dataset)

        tensorboard_logs = {'val_loss': avg_loss}

        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

        return optim

    def prepare_data(self):
        # create dataset
        download_url = 'http://filecremers3.informatik.tu-muenchen.de/~dl4cv/training.zip'
        i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
        data_root = os.path.join(i2dl_exercises_path, "datasets", "facial_keypoints")
        self.train_dataset = FacialKeypointsDataset(
            train=True,
            transform=transforms.ToTensor(),
            root=data_root,
            download_url=download_url
        )
        self.val_dataset = FacialKeypointsDataset(
            train=False,
            transform=transforms.ToTensor(),
            root=data_root,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams["batch_size"])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams["batch_size"])

    def getTestAcc(self, loader=None):
        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch['image'], batch['keypoints']
            X, y = X.to(self.device), y.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
